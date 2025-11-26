import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

# ==========================================
# 1. ASPS 动态采样策略
# ==========================================
def get_contrast_pair_batch(current_epoch, total_epochs, warmup_epochs, feat_sim, device, beta=0.5):
    """
    ASPS (Adaptive Self-Paced Sampling) - 动态硬负样本挖掘
    
    Returns:
        neg_mask: 包含权重的负样本掩码。
                  硬负样本权重 = 1.0
                  普通负样本权重 = 1 - alpha (例如 0.5)
                  正样本位置权重 = 0.0
    """
    batch_size = feat_sim.shape[0]

    # 正样本：对角线
    pos_mask = torch.eye(batch_size, device=device)
    
    # 基础负样本：所有非对角线
    neg_all = 1.0 - pos_mask

    # Warmup 期间不进行硬负样本挖掘，只返回普通负样本掩码
    if current_epoch <= warmup_epochs:
        return pos_mask, neg_all
    
    # 计算训练进度 (warmup 后)
    progress = (current_epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    progress = min(max(progress, 0.0), 1.0)
    
    # 动态计算硬负样本数量
    max_neg_num = batch_size - 1
    k_neg = int(max_neg_num * progress * beta)
    k_neg = max(1, min(k_neg, max_neg_num))
    
    # 基于特征相似度挖掘硬负样本
    feat_sim_masked = feat_sim.clone()
    feat_sim_masked.fill_diagonal_(-1e9)  # 排除自身
    
    # 选择相似度最高的 k 个作为硬负样本
    _, hard_indices = feat_sim_masked.topk(k=k_neg, dim=1, largest=True)
    
    hard_neg_mask = torch.zeros_like(feat_sim, device=device)
    hard_neg_mask.scatter_(1, hard_indices, 1.0)
    
    # 渐进式混合策略
    # 随着训练进行，alpha 增大，硬负样本权重维持 1.0，简单负样本权重(1-alpha)降低
    alpha = min(progress, 0.7)  # 限制 alpha 最大为 0.7，保留至少 0.3 的简单样本权重
    neg_mask = alpha * hard_neg_mask + (1 - alpha) * neg_all
    
    # 确保正样本位置为 0 (不参与负样本计算)
    neg_mask = neg_mask * (1 - pos_mask)
    
    return pos_mask, neg_mask


# ==========================================
# 2. CCL 核心对比模块 - InfoNCE (修复逻辑版)
# ==========================================
class Model_Contrast(nn.Module):
    """
    InfoNCE 风格对比学习模块（支持 ASPS 软权重掩码）
    """
    def __init__(self, hidden_dim, tau=0.1):
        super(Model_Contrast, self).__init__()
        self.tau = tau
        
        # 投影头
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.projector:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None: nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

    def safe_normalize(self, x, dim=1, eps=1e-8):
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        return F.normalize(x + eps, dim=dim)

    def info_nce_loss(self, anchor, positive, neg_mask):
        """
        ⭐ 修复版 InfoNCE Loss
        """
        anchor = self.safe_normalize(anchor)
        positive = self.safe_normalize(positive)
        
        batch_size = anchor.shape[0]
        device = anchor.device
        
        # 1. 计算相似度矩阵
        logits = torch.mm(anchor, positive.t()) / self.tau  # [B, B]
        
        # 2. 提取正样本 (对角线)
        pos_sim = torch.diag(logits)  # [B]
        
        # 3. 只对非对角线（负样本）应用权重
        #    将对角线完全置为 -inf，避免正样本进入分母
        neg_logits = logits.clone()
        neg_logits.fill_diagonal_(-float('inf'))  # 完全排除正样本
        
        # 4. 应用软权重: weight * exp(x) = exp(x + log(weight))
        #    对于 weight=0 的位置，log(0)=-inf，exp(-inf)=0，正确忽略
        safe_mask = neg_mask.clone()
        safe_mask.fill_diagonal_(0)  # 确保对角线为0
        log_weights = torch.log(safe_mask + 1e-9)
        neg_logits_weighted = neg_logits + log_weights
        
        # 5. 分母 = exp(pos) + sum(weighted_neg)
        #    使用 logsumexp 保持数值稳定
        # 先计算 logsumexp(neg)，再与 pos 合并
        neg_logsumexp = torch.logsumexp(neg_logits_weighted, dim=1)  # [B]
        
        # log(exp(pos) + exp(neg_sum)) = logsumexp([pos, neg_sum])
        denominator = torch.logsumexp(
            torch.stack([pos_sim, neg_logsumexp], dim=1), dim=1
        )
        
        # 6. Loss = -log(exp(pos) / denom) = -(pos - log(denom))
        loss = -(pos_sim - denominator).mean()
        
        return loss

    def forward(self, v1_embs, v2_embs, fused_embs, pos_mask, neg_mask):
        # 投影
        z1 = self.projector(v1_embs)
        z2 = self.projector(v2_embs)
        z_fused = self.projector(fused_embs.detach()) # Teacher detach
        
        # 双向蒸馏 Loss
        loss_v1 = self.info_nce_loss(z1, z_fused, neg_mask)
        loss_v2 = self.info_nce_loss(z2, z_fused, neg_mask)
        
        return 0.5 * (loss_v1 + loss_v2)


# ==========================================
# 3. 主模型 (AttnFusionGCNNet)
# ==========================================
class AttnFusionGCNNet(torch.nn.Module):
    def __init__(self, n_output=1, n_filters=32, embed_dim=64, num_features_xd=78,
                 num_features_smile=66, num_features_xt=25, output_dim=128, dropout=0.2,
                 temperature=0.1):
        super(AttnFusionGCNNet, self).__init__()

        self.n_output = n_output
        self.output_dim = output_dim
        self.temperature = temperature

        # Embedding
        self.max_smile_idx = num_features_smile
        self.max_target_idx = num_features_xt
        self.smile_embed = nn.Embedding(num_features_smile + 1, embed_dim)

        # ============ Drug Encoders ============
        self.conv_xd_11 = nn.Conv1d(embed_dim, n_filters, kernel_size=3, padding=1)
        self.conv_xd_12 = nn.Conv1d(n_filters, n_filters * 2, kernel_size=3, padding=1)
        self.conv_xd_21 = nn.Conv1d(embed_dim, n_filters, kernel_size=2, padding=1)
        self.conv_xd_22 = nn.Conv1d(n_filters, n_filters * 2, kernel_size=2, padding=1)
        self.conv_xd_31 = nn.Conv1d(embed_dim, n_filters, kernel_size=1, padding=1)
        self.conv_xd_32 = nn.Conv1d(n_filters, n_filters * 2, kernel_size=1, padding=1)

        self.fc_smiles = nn.Linear(n_filters * 2, output_dim)

        # Drug Fingerprint
        self.rdkit_descriptor_dim = 210
        self.rdkit_fingerprint_dim = 136
        self.maccs_dim = 166
        self.morgan_dim = 512
        self.combined_dim = 1024
        self.attention_rdkit = nn.Linear(self.rdkit_descriptor_dim, self.rdkit_descriptor_dim)
        self.attention_maccs = nn.Linear(self.maccs_dim, self.maccs_dim)
        self.drug_fp_transform = nn.Linear(self.combined_dim, output_dim)

        # Drug Fusion
        self.drug_attn = nn.MultiheadAttention(output_dim, num_heads=8, batch_first=True, dropout=0.1)
        self.layer_norm_drug = nn.LayerNorm(output_dim)
        self.relu = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(dropout)
        self.fusion_drug = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout)
        )
        
        # ⭐ 修复：动态计算 reduce 通道数，防止 n_filters 变化时报错
        self.conv_reduce_smiles = nn.Conv1d(output_dim * 3, output_dim, kernel_size=1)

        # ============ miRNA Encoders ============
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        
        self.conv_xt_11 = nn.Conv1d(embed_dim, n_filters, kernel_size=4, padding=2)
        self.conv_xt_12 = nn.Conv1d(n_filters, n_filters * 2, kernel_size=4, padding=2)
        self.conv_xt_21 = nn.Conv1d(embed_dim, n_filters, kernel_size=3, padding=1)
        self.conv_xt_22 = nn.Conv1d(n_filters, n_filters * 2, kernel_size=3, padding=1)
        self.conv_xt_31 = nn.Conv1d(embed_dim, n_filters, kernel_size=2, padding=1)
        self.conv_xt_32 = nn.Conv1d(n_filters, n_filters * 2, kernel_size=2, padding=1)

        # ⭐ 修复：动态计算 reduce 通道数
        total_xt_channels = n_filters * 2 * 3
        self.conv_reduce_xt = nn.Conv1d(total_xt_channels, output_dim, kernel_size=1)

        # Matrix CNN
        self.conv_matrix_1 = nn.Conv2d(1, n_filters, kernel_size=3, padding=1)
        self.conv_matrix_2 = nn.Conv2d(n_filters, n_filters * 2, kernel_size=3, padding=1)
        self.conv_matrix_3 = nn.Conv2d(n_filters * 2, n_filters * 4, kernel_size=3, padding=1)
        matrix_fc_dim = n_filters * 4 * 4 * 4
        self.fc_matrix_1 = nn.Linear(matrix_fc_dim, 256)
        self.fc_matrix_2 = nn.Linear(256, output_dim)

        # miRNA Fusion
        self.mirna_attn = nn.MultiheadAttention(output_dim, num_heads=8, batch_first=True, dropout=0.05)
        self.layer_norm_mirna = nn.LayerNorm(output_dim)

        # ============ 对比学习模块 ============
        self.contrast_drug = Model_Contrast(hidden_dim=output_dim, tau=temperature)
        self.contrast_mirna = Model_Contrast(hidden_dim=output_dim, tau=temperature)

        # Final
        self.fc1 = nn.Linear(output_dim * 2, 256)
        self.out = nn.Linear(256, self.n_output)
        self.ac = nn.Sigmoid()

    def _safe_tensor(self, x):
        return torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

    def process_drug_fingerprints(self, rdkit_desc, rdkit_fp, maccs_fp, morgan_fp):
        # 确保输入是 2D
        if len(rdkit_desc.shape) == 1: rdkit_desc = rdkit_desc.unsqueeze(0)
        if len(rdkit_fp.shape) == 1: rdkit_fp = rdkit_fp.unsqueeze(0)
        if len(maccs_fp.shape) == 1: maccs_fp = maccs_fp.unsqueeze(0)
        if len(morgan_fp.shape) == 1: morgan_fp = morgan_fp.unsqueeze(0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            if len(rdkit_desc.shape) > 2: rdkit_desc = rdkit_desc.mean(dim=1)
            if len(rdkit_fp.shape) > 2: rdkit_fp = rdkit_fp.mean(dim=1)
            if len(maccs_fp.shape) > 2: maccs_fp = maccs_fp.mean(dim=1)
            if len(morgan_fp.shape) > 2: morgan_fp = morgan_fp.mean(dim=1)

        rdkit_desc = self._safe_tensor(rdkit_desc)
        rdkit_fp = self._safe_tensor(rdkit_fp)
        maccs_fp = self._safe_tensor(maccs_fp)
        morgan_fp = self._safe_tensor(morgan_fp)

        attn_rdkit = F.softmax(self.attention_rdkit(rdkit_desc), dim=-1)
        rdkit_prime = rdkit_desc * attn_rdkit

        attn_maccs = F.softmax(self.attention_maccs(maccs_fp), dim=-1)
        maccs_prime = maccs_fp * attn_maccs

        combined = torch.cat([rdkit_prime, maccs_prime, rdkit_fp, morgan_fp], dim=-1)
        features = self.dropout(self.relu(self.drug_fp_transform(combined)))
        return self._safe_tensor(features)

    def forward(self, data, current_epoch=0, total_epochs=100, warmup_epochs=5, 
                return_contrastive_loss=True, **kwargs):
        """
        前向传播 - 支持 ASPS 动态采样
        """
        # ============= 数据预处理 =============
        drugsmile = data.seqdrug
        target = data.target
        target_matrix = data.target_matrix
        
        if drugsmile.dtype in [torch.float32, torch.float64]: drugsmile = drugsmile.long()
        if target.dtype in [torch.float32, torch.float64]: target = target.long()
            
        drugsmile = torch.clamp(drugsmile, 0, self.max_smile_idx)
        target = torch.clamp(target, 0, self.max_target_idx)
        batch_size = drugsmile.shape[0]
        device = drugsmile.device

        # 指纹特征
        rdkit_desc = self._safe_tensor(data.rdkit_descriptor.view(batch_size, self.rdkit_descriptor_dim))
        rdkit_fp = self._safe_tensor(data.rdkit_fingerprint.view(batch_size, self.rdkit_fingerprint_dim))
        maccs_fp = self._safe_tensor(data.maccs_fingerprint.view(batch_size, self.maccs_dim))
        morgan_fp = self._safe_tensor(data.morgan_fingerprint.view(batch_size, self.morgan_dim))
        target_matrix = self._safe_tensor(target_matrix)

        # ============= Drug Processing =============
        drug_mol_features = self.process_drug_fingerprints(rdkit_desc, rdkit_fp, maccs_fp, morgan_fp)

        embedded_smile = self.smile_embed(drugsmile).permute(0, 2, 1)
        
        conv_xd1 = F.max_pool1d(self.relu(self.dropout(self.conv_xd_11(embedded_smile))), 2)
        conv_xd1 = F.max_pool1d(self.relu(self.conv_xd_12(conv_xd1)), conv_xd1.size(2)).squeeze(2)
        
        conv_xd2 = F.max_pool1d(self.relu(self.dropout(self.conv_xd_21(embedded_smile))), 2)
        conv_xd2 = F.max_pool1d(self.relu(self.dropout(self.conv_xd_22(conv_xd2))), conv_xd2.size(2)).squeeze(2)
        
        conv_xd3 = F.max_pool1d(self.relu(self.dropout(self.conv_xd_31(embedded_smile))), 2)
        conv_xd3 = F.max_pool1d(self.relu(self.conv_xd_32(conv_xd3)), conv_xd3.size(2)).squeeze(2)

        conv_xd = torch.cat([self.fc_smiles(conv_xd1), self.fc_smiles(conv_xd2), self.fc_smiles(conv_xd3)], dim=1)
        drug_seq_features = self._safe_tensor(self.conv_reduce_smiles(conv_xd.unsqueeze(2)).squeeze(2))

        # Drug Fusion
        # Query: Seq, Key/Value: Mol
        attn_out, _ = self.drug_attn(drug_seq_features.unsqueeze(1), drug_mol_features.unsqueeze(1), drug_mol_features.unsqueeze(1))
        drug_fused = self.layer_norm_drug(self._safe_tensor(attn_out.squeeze(1)) + drug_seq_features)
        drug_features = self._safe_tensor(self.fusion_drug(torch.cat([drug_fused, drug_mol_features], dim=1)))

        # ============= miRNA Processing =============
        embedded_xt = self.embedding_xt(target).permute(0, 2, 1)
        
        conv_xt1 = F.max_pool1d(self.relu(self.conv_xt_12(self.relu(self.dropout(self.conv_xt_11(embedded_xt))))), embedded_xt.size(2)).squeeze(2)
        conv_xt2 = F.max_pool1d(self.relu(self.conv_xt_22(self.relu(self.dropout(self.conv_xt_21(embedded_xt))))), embedded_xt.size(2)).squeeze(2)
        conv_xt3 = F.max_pool1d(self.relu(self.conv_xt_32(F.max_pool1d(self.relu(self.dropout(self.conv_xt_31(embedded_xt))), 2))), embedded_xt.size(2) // 2).squeeze(2)

        mirna_seq_features = self._safe_tensor(self.conv_reduce_xt(torch.cat([conv_xt1, conv_xt2, conv_xt3], dim=1).unsqueeze(2)).squeeze(2))

        # Matrix
        if len(target_matrix.shape) == 3:
            target_matrix = target_matrix.unsqueeze(1)
        
        mat = F.max_pool2d(self.relu(self.conv_matrix_1(target_matrix)), 2)
        mat = F.max_pool2d(self.relu(self.conv_matrix_2(mat)), 2)
        mat = self.dropout(self.relu(self.conv_matrix_3(mat)))
        mirna_cgr_features = self._safe_tensor(self.fc_matrix_2(self.dropout(self.relu(self.fc_matrix_1(mat.view(batch_size, -1))))))

        # miRNA Fusion
        attn_m, _ = self.mirna_attn(mirna_seq_features.unsqueeze(1), mirna_cgr_features.unsqueeze(1), mirna_cgr_features.unsqueeze(1))
        mirna_features = self._safe_tensor(self.relu(self.layer_norm_mirna(self._safe_tensor(attn_m.squeeze(1)) + mirna_seq_features)))

        # ============= Final Prediction =============
        xc = self.dropout(self.relu(self.fc1(torch.cat([drug_features, mirna_features], dim=1))))
        out = torch.clamp(self.out(xc), -10, 10)
        out = torch.clamp(self.ac(out), 1e-7, 1 - 1e-7)

        # ============= ASPS + InfoNCE 对比损失 =============
        if return_contrastive_loss:
            # 计算融合特征的相似度矩阵 (用于 ASPS 挖掘)
            mirna_fused_norm = F.normalize(mirna_features.detach(), dim=1)
            drug_fused_norm = F.normalize(drug_features.detach(), dim=1)
            
            mirna_sim = torch.mm(mirna_fused_norm, mirna_fused_norm.t())
            drug_sim = torch.mm(drug_fused_norm, drug_fused_norm.t())
            
            # ASPS 动态采样 (生成软权重掩码)
            pos_mask_m, neg_mask_m = get_contrast_pair_batch(
                current_epoch, total_epochs, warmup_epochs, mirna_sim, device
            )
            pos_mask_d, neg_mask_d = get_contrast_pair_batch(
                current_epoch, total_epochs, warmup_epochs, drug_sim, device
            )
            
            # InfoNCE Loss (使用修复后的逻辑)
            loss_mirna = self.contrast_mirna(mirna_seq_features, mirna_cgr_features, mirna_features, pos_mask_m, neg_mask_m)
            loss_drug = self.contrast_drug(drug_seq_features, drug_mol_features, drug_features, pos_mask_d, neg_mask_d)

            return out, {'contrastive_mirna': loss_mirna, 'contrastive_drug': loss_drug}

        return out