import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings


# ==========================================
# 1. CCL 核心对比模块 (Model_Contrast) - 修复版
# ==========================================
class Model_Contrast(nn.Module):
    def __init__(self, hidden_dim, tau, lam):
        super(Model_Contrast, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau = tau
        self.lam = lam
        # 初始化权重
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        """计算余弦相似度矩阵"""
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())

        # 修复: 先计算余弦相似度，再除以温度，最后 exp
        cos_sim = dot_numerator / (dot_denominator + 1e-8)
        sim_matrix = torch.exp(cos_sim / self.tau)
        return sim_matrix

    def forward(self, v1_embs, v2_embs, pos=None, neg=None):
        """
        修复: InfoNCE 损失应该按样本计算，而不是全局求和

        Args:
            v1_embs: View 1 embeddings [batch_size, hidden_dim]
            v2_embs: View 2 embeddings [batch_size, hidden_dim]
            pos: Positive mask [batch_size, batch_size]
            neg: Negative mask [batch_size, batch_size]
        """
        v1_embs = self.proj(v1_embs)
        v2_embs = self.proj(v2_embs)

        # 计算相似度矩阵
        matrix_1to2 = self.sim(v1_embs, v2_embs)

        # 应用掩码: 正样本和负样本的相似度
        pos_sim = (matrix_1to2 * pos).sum(dim=1)  # [batch_size]
        neg_sim = (matrix_1to2 * neg).sum(dim=1)  # [batch_size]

        # 修复: 按样本计算 InfoNCE 损失
        # Loss = -log(exp(pos) / (exp(pos) + exp(neg)))
        #      = -log(pos / (pos + neg))
        loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8) + 1e-8)

        return loss.mean()


# ==========================================
# 2. ASPS 动态采样策略 (修复版)
# ==========================================
def get_contrast_pair_batch(args, feat_sim, device):
    """
    ASPS (Adaptive Self-Paced Sampling) - 修复版

    改进点:
    1. 延迟硬负样本挖掘启动时间（与 Warmup 配合）
    2. 渐进式混合策略（而非硬切换）
    3. 更合理的 k_neg 计算
    """
    batch_size = feat_sim.shape[0]

    # 解析参数
    current_epoch = args.current_epoch if hasattr(args, 'current_epoch') else args['current_epoch']
    total_epoch = args.epochs if hasattr(args, 'epochs') else args['epochs']
    beta = args.beta if hasattr(args, 'beta') else 0.5
    warmup_epochs = args.get('warmup_epochs', 5)  # 新增: 支持 warmup 参数

    # 1. 基础正样本 (Positive): 对角线
    pos = torch.eye(batch_size).to(device)

    # 2. 基础负样本 (Negative): 所有非对角线
    neg_all = torch.ones_like(pos) - pos

    # 3. 修复: ASPS 动态采样策略
    max_neg_num = batch_size - 1

    # 修复: 在 Warmup 期间不进行硬负样本挖掘
    if current_epoch <= warmup_epochs:
        # Warmup 阶段: 使用所有负样本
        neg = neg_all
    else:
        # 修复: 计算相对于 warmup 后的进度
        progress = (current_epoch - warmup_epochs) / (total_epoch - warmup_epochs)
        progress = max(0.0, min(1.0, progress))  # 限制在 [0, 1]

        # 修复: 使用二次增长，早期温和增加
        k_neg = int(max_neg_num * (progress ** 1.5) * beta)
        k_neg = max(1, min(k_neg, max_neg_num))  # 限制在 [1, max_neg_num]

        # 利用特征相似度挖掘困难样本
        feat_sim_masked = feat_sim.clone()
        feat_sim_masked.fill_diagonal_(-9e15)  # 排除自己

        # 选出相似度最高的 K 个作为 "Hard Negatives"
        vals, indices = feat_sim_masked.topk(k=k_neg, dim=1, largest=True)

        hard_neg_mask = torch.zeros_like(feat_sim).to(device)
        hard_neg_mask.scatter_(1, indices, 1)

        # 修复: 渐进式混合策略
        # 早期: 主要用全部负样本
        # 后期: 逐渐增加困难负样本权重
        alpha = min(progress * 2, 1.0)  # 0 -> 1 的权重
        neg = alpha * hard_neg_mask + (1 - alpha) * neg_all

    # 确保正负不重叠
    neg = neg * (1 - pos)

    return pos, neg


# ==========================================
# 3. 主模型 (AttnFusionGCNNet) - 修复版
# ==========================================
class AttnFusionGCNNet(torch.nn.Module):
    def __init__(self, n_output=1, n_filters=32, embed_dim=64, num_features_xd=78,
                 num_features_smile=66, num_features_xt=25, output_dim=128, dropout=0.2,
                 contrastive_dim=128, temperature=0.1, lam=0.5):
        super(AttnFusionGCNNet, self).__init__()

        self.n_output = n_output
        self.output_dim = output_dim
        self.contrastive_dim = contrastive_dim

        # Embedding 参数
        self.max_smile_idx = num_features_smile
        self.max_target_idx = num_features_xt
        self.smile_embed = nn.Embedding(num_features_smile + 1, embed_dim)

        # ============ Drug Encoders ============
        # CNN Branch 1
        self.conv_xd_11 = nn.Conv1d(embed_dim, out_channels=n_filters, kernel_size=3, padding=1)
        self.conv_xd_12 = nn.Conv1d(n_filters, out_channels=n_filters * 2, kernel_size=3, padding=1)
        # CNN Branch 2
        self.conv_xd_21 = nn.Conv1d(embed_dim, out_channels=n_filters, kernel_size=2, padding=1)
        self.conv_xd_22 = nn.Conv1d(n_filters, out_channels=n_filters * 2, kernel_size=2, padding=1)
        # CNN Branch 3
        self.conv_xd_31 = nn.Conv1d(embed_dim, out_channels=n_filters, kernel_size=1, padding=1)
        self.conv_xd_32 = nn.Conv1d(n_filters, out_channels=n_filters * 2, kernel_size=1, padding=1)

        self.fc_smiles = nn.Linear(n_filters * 2, output_dim)

        # Drug Fingerprint components
        self.rdkit_descriptor_dim = 210
        self.rdkit_fingerprint_dim = 136
        self.maccs_dim = 166
        self.morgan_dim = 512
        self.combined_dim = 1024
        self.attention_rdkit_descriptor = nn.Linear(self.rdkit_descriptor_dim, self.rdkit_descriptor_dim)
        self.attention_maccs = nn.Linear(self.maccs_dim, self.maccs_dim)
        self.drug_fingerprint_transform = nn.Linear(self.combined_dim, output_dim)

        # Drug Feature Fusion
        self.drug_attn = nn.MultiheadAttention(embed_dim=output_dim, num_heads=8, batch_first=True, dropout=0.1)
        self.layer_norm_drug = nn.LayerNorm(output_dim, eps=1e-3)
        self.relu = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(dropout)

        self.fusion_drug_final = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            self.relu,
            self.dropout
        )

        self.conv_reduce_smiles = nn.Conv1d(in_channels=output_dim * 3, out_channels=output_dim, kernel_size=1)
        self.conv_reduce_xt = nn.Conv1d(in_channels=192, out_channels=output_dim, kernel_size=1)

        # ============ miRNA Encoders ============
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)

        # miRNA Sequence CNNs
        self.conv_xt_11 = nn.Conv1d(embed_dim, out_channels=n_filters, kernel_size=4, padding=2)
        self.conv_xt_12 = nn.Conv1d(n_filters, out_channels=n_filters * 2, kernel_size=4, padding=2)
        self.conv_xt_21 = nn.Conv1d(embed_dim, out_channels=n_filters, kernel_size=3, padding=1)
        self.conv_xt_22 = nn.Conv1d(n_filters, out_channels=n_filters * 2, kernel_size=3, padding=1)
        self.conv_xt_31 = nn.Conv1d(embed_dim, out_channels=n_filters, kernel_size=2, padding=1)
        self.conv_xt_32 = nn.Conv1d(n_filters, out_channels=n_filters * 2, kernel_size=2, padding=1)

        # miRNA Matrix CNNs
        self.conv_matrix_1 = nn.Conv2d(1, n_filters, kernel_size=3, padding=1)
        self.conv_matrix_2 = nn.Conv2d(n_filters, n_filters * 2, kernel_size=3, padding=1)
        self.conv_matrix_3 = nn.Conv2d(n_filters * 2, n_filters * 4, kernel_size=3, padding=1)
        self.fc_matrix_1 = nn.Linear(n_filters * 4 * 4 * 4, 256)
        self.fc_matrix_2 = nn.Linear(256, output_dim)

        self.mirna_attn = nn.MultiheadAttention(embed_dim=output_dim, num_heads=8, batch_first=True, dropout=0.05)
        self.layer_norm_mirna = nn.LayerNorm(output_dim, eps=1e-3)

        # ============ 对比学习模块 (修复版) ============
        self.contrast_drug = Model_Contrast(hidden_dim=output_dim, tau=temperature, lam=lam)
        self.contrast_mirna = Model_Contrast(hidden_dim=output_dim, tau=temperature, lam=lam)

        # Final layers
        self.fc1 = nn.Linear(output_dim * 2, 256)
        self.out = nn.Linear(256, self.n_output)
        self.ac = nn.Sigmoid()

    def process_drug_fingerprints(self, rdkit_descriptor, rdkit_fingerprint, maccs_fingerprint, morgan_fingerprint):
        """Process drug fingerprint features with self-attention"""
        if len(rdkit_descriptor.shape) == 1: rdkit_descriptor = rdkit_descriptor.unsqueeze(0)
        if len(rdkit_fingerprint.shape) == 1: rdkit_fingerprint = rdkit_fingerprint.unsqueeze(0)
        if len(maccs_fingerprint.shape) == 1: maccs_fingerprint = maccs_fingerprint.unsqueeze(0)
        if len(morgan_fingerprint.shape) == 1: morgan_fingerprint = morgan_fingerprint.unsqueeze(0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            if len(rdkit_descriptor.shape) > 2: rdkit_descriptor = rdkit_descriptor.mean(dim=1)
            if len(rdkit_fingerprint.shape) > 2: rdkit_fingerprint = rdkit_fingerprint.mean(dim=1)
            if len(maccs_fingerprint.shape) > 2: maccs_fingerprint = maccs_fingerprint.mean(dim=1)
            if len(morgan_fingerprint.shape) > 2: morgan_fingerprint = morgan_fingerprint.mean(dim=1)

        attention_weights_rdkit = self.attention_rdkit_descriptor(rdkit_descriptor)
        attention_weights_rdkit = F.softmax(attention_weights_rdkit, dim=-1)
        rdkit_descriptor_prime = rdkit_descriptor * attention_weights_rdkit

        attention_weights_maccs = self.attention_maccs(maccs_fingerprint)
        attention_weights_maccs = F.softmax(attention_weights_maccs, dim=-1)
        maccs_prime = maccs_fingerprint * attention_weights_maccs

        combined_features = torch.cat([
            rdkit_descriptor_prime,
            maccs_prime,
            rdkit_fingerprint,
            morgan_fingerprint
        ], dim=-1)

        drug_features = self.drug_fingerprint_transform(combined_features)
        drug_features = self.relu(drug_features)
        drug_features = self.dropout(drug_features)
        drug_features = torch.nan_to_num(drug_features, nan=0.0, posinf=0.0, neginf=0.0)
        return drug_features

    def forward(self, data, current_epoch=0, total_epochs=100, warmup_epochs=5, return_contrastive_loss=True):
        """
        Forward pass with CCL-ASPS Logic (修复版)

        新增参数:
            warmup_epochs: Warmup 轮数，用于配合 ASPS
        """
        # ============= Data Loading & Preprocessing =============
        rdkit_fingerprint = data.rdkit_fingerprint
        rdkit_descriptor = data.rdkit_descriptor
        maccs_fingerprint = data.maccs_fingerprint
        morgan_fingerprint = data.morgan_fingerprint
        drugsmile = data.seqdrug
        target = data.target
        target_matrix = data.target_matrix if hasattr(data, 'target_matrix') else None

        if target_matrix is None: raise ValueError("target_matrix is None.")

        if drugsmile.dtype == torch.float32 or drugsmile.dtype == torch.float64: drugsmile = drugsmile.long()
        if target.dtype == torch.float32 or target.dtype == torch.float64: target = target.long()
        drugsmile = torch.clamp(drugsmile, 0, self.max_smile_idx)
        target = torch.clamp(target, 0, self.max_target_idx)
        batch_size = drugsmile.shape[0]

        rdkit_descriptor = rdkit_descriptor.view(batch_size, self.rdkit_descriptor_dim)
        rdkit_fingerprint = rdkit_fingerprint.view(batch_size, self.rdkit_fingerprint_dim)
        maccs_fingerprint = maccs_fingerprint.view(batch_size, self.maccs_dim)
        morgan_fingerprint = morgan_fingerprint.view(batch_size, self.morgan_dim)

        # NaN Handling
        rdkit_descriptor = torch.nan_to_num(rdkit_descriptor, nan=0.0)
        rdkit_fingerprint = torch.nan_to_num(rdkit_fingerprint, nan=0.0)
        maccs_fingerprint = torch.nan_to_num(maccs_fingerprint, nan=0.0)
        morgan_fingerprint = torch.nan_to_num(morgan_fingerprint, nan=0.0)
        target_matrix = torch.nan_to_num(target_matrix, nan=0.0)

        # ============= Drug Processing =============

        # 1. Drug Fingerprint Processing (View 1)
        fingerprint_features = self.process_drug_fingerprints(
            rdkit_descriptor, rdkit_fingerprint, maccs_fingerprint, morgan_fingerprint
        )
        drug_mol_features = fingerprint_features

        # 2. SMILES Sequence Processing (View 2)
        embedded_smile = self.smile_embed(drugsmile).permute(0, 2, 1)

        # Branch 1
        conv_xd1 = self.conv_xd_11(embedded_smile)
        conv_xd1 = self.relu(conv_xd1)
        conv_xd1 = self.dropout(conv_xd1)
        conv_xd1 = F.max_pool1d(conv_xd1, kernel_size=2)
        conv_xd1 = self.conv_xd_12(conv_xd1)
        conv_xd1 = self.relu(conv_xd1)
        conv_xd1 = F.max_pool1d(conv_xd1, conv_xd1.size(2)).squeeze(2)

        # Branch 2
        conv_xd2 = self.conv_xd_21(embedded_smile)
        conv_xd2 = self.relu(conv_xd2)
        conv_xd2 = self.dropout(conv_xd2)
        conv_xd2 = F.max_pool1d(conv_xd2, kernel_size=2)
        conv_xd2 = self.conv_xd_22(conv_xd2)
        conv_xd2 = self.relu(conv_xd2)
        conv_xd2 = self.dropout(conv_xd2)
        conv_xd2 = F.max_pool1d(conv_xd2, conv_xd2.size(2)).squeeze(2)

        # Branch 3
        conv_xd3 = self.conv_xd_31(embedded_smile)
        conv_xd3 = self.relu(conv_xd3)
        conv_xd3 = self.dropout(conv_xd3)
        conv_xd3 = F.max_pool1d(conv_xd3, kernel_size=2)
        conv_xd3 = self.conv_xd_32(conv_xd3)
        conv_xd3 = self.relu(conv_xd3)
        conv_xd3 = F.max_pool1d(conv_xd3, conv_xd3.size(2)).squeeze(2)

        # Combine branches
        conv_xd1 = self.fc_smiles(conv_xd1)
        conv_xd2 = self.fc_smiles(conv_xd2)
        conv_xd3 = self.fc_smiles(conv_xd3)

        conv_xd = torch.cat((conv_xd1, conv_xd2, conv_xd3), dim=1).unsqueeze(1).permute(0, 2, 1)
        conv_xd = self.conv_reduce_smiles(conv_xd).squeeze(2)
        conv_xd = torch.nan_to_num(conv_xd, nan=0.0)
        drug_seq_features = conv_xd

        # 3. Drug Fusion (Fused View)
        smiles_unsq = conv_xd.unsqueeze(1)
        fingerprint_unsq = fingerprint_features.unsqueeze(1)

        attn_out, _ = self.drug_attn(query=smiles_unsq, key=fingerprint_unsq, value=fingerprint_unsq)
        attn_out = torch.nan_to_num(attn_out.squeeze(1), nan=0.0)

        residual_in_drug = attn_out + conv_xd
        drug_features_attn = self.layer_norm_drug(residual_in_drug)

        drug_concat = torch.cat([drug_features_attn, fingerprint_features], dim=1)
        drug_features = self.fusion_drug_final(drug_concat)

        # ============= miRNA Processing =============

        # 1. miRNA Sequence Processing (View 1)
        embedded_xt = self.embedding_xt(target).permute(0, 2, 1)

        # Branch 1
        conv_xt1 = self.conv_xt_11(embedded_xt)
        conv_xt1 = self.relu(conv_xt1)
        conv_xt1 = self.dropout(conv_xt1)
        conv_xt1 = self.conv_xt_12(conv_xt1)
        conv_xt1 = self.relu(conv_xt1)
        conv_xt1 = F.max_pool1d(conv_xt1, conv_xt1.size(2)).squeeze(2)

        # Branch 2
        conv_xt2 = self.conv_xt_21(embedded_xt)
        conv_xt2 = self.relu(conv_xt2)
        conv_xt2 = self.dropout(conv_xt2)
        conv_xt2 = self.conv_xt_22(conv_xt2)
        conv_xt2 = self.relu(conv_xt2)
        conv_xt2 = F.max_pool1d(conv_xt2, conv_xt2.size(2)).squeeze(2)

        # Branch 3
        conv_xt3 = self.conv_xt_31(embedded_xt)
        conv_xt3 = self.relu(conv_xt3)
        conv_xt3 = self.dropout(conv_xt3)
        conv_xt3 = F.max_pool1d(conv_xt3, kernel_size=2)
        conv_xt3 = self.conv_xt_32(conv_xt3)
        conv_xt3 = self.relu(conv_xt3)
        conv_xt3 = F.max_pool1d(conv_xt3, conv_xt3.size(2)).squeeze(2)

        # Combine branches
        conv_xt = torch.cat((conv_xt1, conv_xt2, conv_xt3), dim=1).unsqueeze(2)
        conv_xt = self.conv_reduce_xt(conv_xt).squeeze(2)
        conv_xt = torch.nan_to_num(conv_xt, nan=0.0)
        mirna_seq_features = conv_xt

        # 2. miRNA Matrix Processing (View 2)
        if len(target_matrix.shape) == 3: target_matrix = target_matrix.unsqueeze(1)

        matrix_feat = self.conv_matrix_1(target_matrix)
        matrix_feat = F.max_pool2d(self.relu(matrix_feat), kernel_size=2)

        matrix_feat = self.conv_matrix_2(matrix_feat)
        matrix_feat = F.max_pool2d(self.relu(matrix_feat), kernel_size=2)

        matrix_feat = self.conv_matrix_3(matrix_feat)
        matrix_feat = self.relu(matrix_feat)
        matrix_feat = self.dropout(matrix_feat)

        matrix_feat = matrix_feat.view(matrix_feat.size(0), -1)
        matrix_feat = self.fc_matrix_1(matrix_feat)
        matrix_feat = self.relu(matrix_feat)
        matrix_feat = self.dropout(matrix_feat)

        matrix_feat = self.fc_matrix_2(matrix_feat)
        matrix_feat = torch.nan_to_num(matrix_feat, nan=0.0)
        mirna_cgr_features = matrix_feat

        # 3. miRNA Fusion (Fused View)
        xt_unsq = conv_xt.unsqueeze(1)
        mat_unsq = matrix_feat.unsqueeze(1)

        attn_out_m, _ = self.mirna_attn(query=xt_unsq, key=mat_unsq, value=mat_unsq)
        attn_out_m = torch.nan_to_num(attn_out_m.squeeze(1), nan=0.0)

        residual_in_mirna = attn_out_m + conv_xt
        mirna_features = self.layer_norm_mirna(residual_in_mirna)
        mirna_features = self.relu(mirna_features)

        # ============= Final Prediction =============
        xc = torch.cat((drug_features, mirna_features), dim=1)
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        out = self.ac(out)

        # ============= CCL-ASPS 对比损失计算 (修复版) =============
        if return_contrastive_loss:
            # 修复: 统一归一化所有特征
            mirna_seq_norm = F.normalize(mirna_seq_features, dim=1)
            mirna_cgr_norm = F.normalize(mirna_cgr_features, dim=1)
            mirna_fused_norm = F.normalize(mirna_features, dim=1)

            drug_seq_norm = F.normalize(drug_seq_features, dim=1)
            drug_mol_norm = F.normalize(drug_mol_features, dim=1)
            drug_fused_norm = F.normalize(drug_features, dim=1)

            # 封装 ASPS 参数 (新增 warmup_epochs)
            args_sim = {
                'current_epoch': current_epoch,
                'epochs': total_epochs,
                'beta': 0.8,  # 修复: 增加到 0.8，允许后期使用更多困难样本
                'warmup_epochs': warmup_epochs
            }

            # --- miRNA 协作对比 (修复版) ---
            mirna_sim_matrix = torch.mm(mirna_fused_norm, mirna_fused_norm.t())
            pos_mask, neg_mask = get_contrast_pair_batch(args_sim, mirna_sim_matrix, data.target.device)

            # 修复: 使用归一化后的特征
            loss_mirna_seq = self.contrast_mirna(mirna_seq_norm, mirna_fused_norm, pos_mask, neg_mask)
            loss_mirna_cgr = self.contrast_mirna(mirna_cgr_norm, mirna_fused_norm, pos_mask, neg_mask)
            loss_mirna_contrastive = loss_mirna_seq + loss_mirna_cgr

            # --- Drug 协作对比 (修复版) ---
            drug_sim_matrix = torch.mm(drug_fused_norm, drug_fused_norm.t())
            pos_mask_d, neg_mask_d = get_contrast_pair_batch(args_sim, drug_sim_matrix, data.target.device)

            # 修复: 使用归一化后的特征
            loss_drug_seq = self.contrast_drug(drug_seq_norm, drug_fused_norm, pos_mask_d, neg_mask_d)
            loss_drug_mol = self.contrast_drug(drug_mol_norm, drug_fused_norm, pos_mask_d, neg_mask_d)
            loss_drug_contrastive = loss_drug_seq + loss_drug_mol

            loss_dict = {
                'contrastive_mirna': loss_mirna_contrastive,
                'contrastive_drug': loss_drug_contrastive,
            }

            return out, loss_dict

        return out