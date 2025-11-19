import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings


# ==========================================
# CCL 核心对比模块 (保持不变)
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
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        cos_sim = dot_numerator / (dot_denominator + 1e-8)
        sim_matrix = torch.exp(cos_sim / self.tau)
        return sim_matrix

    def forward(self, v1_embs, v2_embs, pos=None, neg=None):
        v1_embs = self.proj(v1_embs)
        v2_embs = self.proj(v2_embs)
        matrix_1to2 = self.sim(v1_embs, v2_embs)
        pos_sim = (matrix_1to2 * pos).sum(dim=1)
        neg_sim = (matrix_1to2 * neg).sum(dim=1)
        loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8) + 1e-8)
        return loss.mean()


# ==========================================
# ASPS 动态采样策略 (保持不变)
# ==========================================
def get_contrast_pair_batch(args, feat_sim, device):
    batch_size = feat_sim.shape[0]
    current_epoch = args.current_epoch if hasattr(args, 'current_epoch') else args['current_epoch']
    total_epoch = args.epochs if hasattr(args, 'epochs') else args['epochs']
    beta = args.beta if hasattr(args, 'beta') else 0.5
    warmup_epochs = args.get('warmup_epochs', 5)

    pos = torch.eye(batch_size).to(device)
    neg_all = torch.ones_like(pos) - pos

    if current_epoch <= warmup_epochs:
        neg = neg_all
    else:
        progress = (current_epoch - warmup_epochs) / (total_epoch - warmup_epochs)
        progress = max(0.0, min(1.0, progress))
        max_neg_num = batch_size - 1
        k_neg = int(max_neg_num * (progress ** 1.5) * beta)
        k_neg = max(1, min(k_neg, max_neg_num))

        feat_sim_masked = feat_sim.clone()
        feat_sim_masked.fill_diagonal_(-9e15)
        vals, indices = feat_sim_masked.topk(k=k_neg, dim=1, largest=True)

        hard_neg_mask = torch.zeros_like(feat_sim).to(device)
        hard_neg_mask.scatter_(1, indices, 1)

        alpha = min(progress * 2, 1.0)
        neg = alpha * hard_neg_mask + (1 - alpha) * neg_all

    neg = neg * (1 - pos)
    return pos, neg


# ==========================================
# 【新增】改进的融合模块
# ==========================================
class BidirectionalCrossAttentionFusion(nn.Module):
    """双向交叉注意力融合 - 推荐使用"""

    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn_forward = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            batch_first=True, dropout=dropout
        )
        self.attn_backward = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            batch_first=True, dropout=dropout
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, feat1, feat2):
        """
        Args:
            feat1, feat2: [batch, embed_dim]
        Returns:
            fused: [batch, embed_dim]
        """
        f1_unsq = feat1.unsqueeze(1)
        f2_unsq = feat2.unsqueeze(1)

        # 双向注意力
        attn_1to2, _ = self.attn_forward(query=f1_unsq, key=f2_unsq, value=f2_unsq)
        attn_2to1, _ = self.attn_backward(query=f2_unsq, key=f1_unsq, value=f1_unsq)

        attn_1to2 = attn_1to2.squeeze(1)
        attn_2to1 = attn_2to1.squeeze(1)

        # 残差融合
        fused = feat1 + feat2 + attn_1to2 + attn_2to1
        fused = self.layer_norm(fused)

        # FFN增强
        fused = fused + self.ffn(fused)
        fused = self.layer_norm(fused)

        return fused


# ==========================================
# 主模型 - 改进版
# ==========================================
class AttnFusionGCNNet(torch.nn.Module):
    def __init__(self, n_output=1, n_filters=32, embed_dim=64, num_features_xd=78,
                 num_features_smile=66, num_features_xt=25, output_dim=128, dropout=0.2,
                 contrastive_dim=128, temperature=0.1, lam=0.5, use_improved_fusion=True):
        """
        Args:
            use_improved_fusion: True 使用改进的双向注意力, False 使用原版单向注意力
        """
        super(AttnFusionGCNNet, self).__init__()

        self.n_output = n_output
        self.output_dim = output_dim
        self.contrastive_dim = contrastive_dim
        self.use_improved_fusion = use_improved_fusion

        # Embedding 参数
        self.max_smile_idx = num_features_smile
        self.max_target_idx = num_features_xt
        self.smile_embed = nn.Embedding(num_features_smile + 1, embed_dim)

        # ============ Drug Encoders ============
        self.conv_xd_11 = nn.Conv1d(embed_dim, out_channels=n_filters, kernel_size=3, padding=1)
        self.conv_xd_12 = nn.Conv1d(n_filters, out_channels=n_filters * 2, kernel_size=3, padding=1)
        self.conv_xd_21 = nn.Conv1d(embed_dim, out_channels=n_filters, kernel_size=2, padding=1)
        self.conv_xd_22 = nn.Conv1d(n_filters, out_channels=n_filters * 2, kernel_size=2, padding=1)
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

        # ============ 【改进点1】Drug 特征融合 ============
        if use_improved_fusion:
            # 使用双向交叉注意力
            self.drug_fusion = BidirectionalCrossAttentionFusion(
                embed_dim=output_dim, num_heads=8, dropout=dropout
            )
        else:
            # 使用原版单向注意力
            self.drug_attn = nn.MultiheadAttention(
                embed_dim=output_dim, num_heads=8, batch_first=True, dropout=0.1
            )
            self.layer_norm_drug = nn.LayerNorm(output_dim, eps=1e-3)
            self.fusion_drug_final = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.LeakyReLU(0.01),
                nn.Dropout(dropout)
            )

        self.relu = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(dropout)

        self.conv_reduce_smiles = nn.Conv1d(in_channels=output_dim * 3, out_channels=output_dim, kernel_size=1)
        self.conv_reduce_xt = nn.Conv1d(in_channels=192, out_channels=output_dim, kernel_size=1)

        # ============ miRNA Encoders ============
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_11 = nn.Conv1d(embed_dim, out_channels=n_filters, kernel_size=4, padding=2)
        self.conv_xt_12 = nn.Conv1d(n_filters, out_channels=n_filters * 2, kernel_size=4, padding=2)
        self.conv_xt_21 = nn.Conv1d(embed_dim, out_channels=n_filters, kernel_size=3, padding=1)
        self.conv_xt_22 = nn.Conv1d(n_filters, out_channels=n_filters * 2, kernel_size=3, padding=1)
        self.conv_xt_31 = nn.Conv1d(embed_dim, out_channels=n_filters, kernel_size=2, padding=1)
        self.conv_xt_32 = nn.Conv1d(n_filters, out_channels=n_filters * 2, kernel_size=2, padding=1)

        self.conv_matrix_1 = nn.Conv2d(1, n_filters, kernel_size=3, padding=1)
        self.conv_matrix_2 = nn.Conv2d(n_filters, n_filters * 2, kernel_size=3, padding=1)
        self.conv_matrix_3 = nn.Conv2d(n_filters * 2, n_filters * 4, kernel_size=3, padding=1)
        self.fc_matrix_1 = nn.Linear(n_filters * 4 * 4 * 4, 256)
        self.fc_matrix_2 = nn.Linear(256, output_dim)

        # ============ 【改进点2】miRNA 特征融合 ============
        if use_improved_fusion:
            # 使用双向交叉注意力
            self.mirna_fusion = BidirectionalCrossAttentionFusion(
                embed_dim=output_dim, num_heads=8, dropout=dropout
            )
        else:
            # 使用原版单向注意力
            self.mirna_attn = nn.MultiheadAttention(
                embed_dim=output_dim, num_heads=8, batch_first=True, dropout=0.05
            )
            self.layer_norm_mirna = nn.LayerNorm(output_dim, eps=1e-3)

        # ============ 对比学习模块 ============
        self.contrast_drug = Model_Contrast(hidden_dim=output_dim, tau=temperature, lam=lam)
        self.contrast_mirna = Model_Contrast(hidden_dim=output_dim, tau=temperature, lam=lam)

        # Final layers
        self.fc1 = nn.Linear(output_dim * 2, 256)
        self.out = nn.Linear(256, self.n_output)
        self.ac = nn.Sigmoid()

    def process_drug_fingerprints(self, rdkit_descriptor, rdkit_fingerprint, maccs_fingerprint, morgan_fingerprint):
        """Process drug fingerprint features (保持不变)"""
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
            rdkit_descriptor_prime, maccs_prime,
            rdkit_fingerprint, morgan_fingerprint
        ], dim=-1)

        drug_features = self.drug_fingerprint_transform(combined_features)
        drug_features = self.relu(drug_features)
        drug_features = self.dropout(drug_features)
        drug_features = torch.nan_to_num(drug_features, nan=0.0, posinf=0.0, neginf=0.0)
        return drug_features

    def forward(self, data, current_epoch=0, total_epochs=100, warmup_epochs=5, return_contrastive_loss=True):
        # ============= Data Loading & Preprocessing (保持不变) =============
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

        rdkit_descriptor = torch.nan_to_num(rdkit_descriptor, nan=0.0)
        rdkit_fingerprint = torch.nan_to_num(rdkit_fingerprint, nan=0.0)
        maccs_fingerprint = torch.nan_to_num(maccs_fingerprint, nan=0.0)
        morgan_fingerprint = torch.nan_to_num(morgan_fingerprint, nan=0.0)
        target_matrix = torch.nan_to_num(target_matrix, nan=0.0)

        # ============= Drug Processing (保持不变) =============
        fingerprint_features = self.process_drug_fingerprints(
            rdkit_descriptor, rdkit_fingerprint, maccs_fingerprint, morgan_fingerprint
        )
        drug_mol_features = fingerprint_features

        embedded_smile = self.smile_embed(drugsmile).permute(0, 2, 1)

        # 三个CNN分支 (保持原有逻辑)
        conv_xd1 = self.conv_xd_11(embedded_smile)
        conv_xd1 = self.relu(conv_xd1)
        conv_xd1 = self.dropout(conv_xd1)
        conv_xd1 = F.max_pool1d(conv_xd1, kernel_size=2)
        conv_xd1 = self.conv_xd_12(conv_xd1)
        conv_xd1 = self.relu(conv_xd1)
        conv_xd1 = F.max_pool1d(conv_xd1, conv_xd1.size(2)).squeeze(2)
        conv_xd1 = self.fc_smiles(conv_xd1)

        conv_xd2 = self.conv_xd_21(embedded_smile)
        conv_xd2 = self.relu(conv_xd2)
        conv_xd2 = self.dropout(conv_xd2)
        conv_xd2 = F.max_pool1d(conv_xd2, kernel_size=2)
        conv_xd2 = self.conv_xd_22(conv_xd2)
        conv_xd2 = self.relu(conv_xd2)
        conv_xd2 = self.dropout(conv_xd2)
        conv_xd2 = F.max_pool1d(conv_xd2, conv_xd2.size(2)).squeeze(2)
        conv_xd2 = self.fc_smiles(conv_xd2)

        conv_xd3 = self.conv_xd_31(embedded_smile)
        conv_xd3 = self.relu(conv_xd3)
        conv_xd3 = self.dropout(conv_xd3)
        conv_xd3 = F.max_pool1d(conv_xd3, kernel_size=2)
        conv_xd3 = self.conv_xd_32(conv_xd3)
        conv_xd3 = self.relu(conv_xd3)
        conv_xd3 = F.max_pool1d(conv_xd3, conv_xd3.size(2)).squeeze(2)
        conv_xd3 = self.fc_smiles(conv_xd3)

        conv_xd = torch.cat((conv_xd1, conv_xd2, conv_xd3), dim=1).unsqueeze(1).permute(0, 2, 1)
        conv_xd = self.conv_reduce_smiles(conv_xd).squeeze(2)
        conv_xd = torch.nan_to_num(conv_xd, nan=0.0)
        drug_seq_features = conv_xd

        # ============ 【改进点1】Drug 特征融合 ============
        if self.use_improved_fusion:
            # 使用双向注意力融合
            drug_features = self.drug_fusion(drug_seq_features, fingerprint_features)
        else:
            # 原版单向注意力
            smiles_unsq = conv_xd.unsqueeze(1)
            fingerprint_unsq = fingerprint_features.unsqueeze(1)
            attn_out, _ = self.drug_attn(query=smiles_unsq, key=fingerprint_unsq, value=fingerprint_unsq)
            attn_out = torch.nan_to_num(attn_out.squeeze(1), nan=0.0)
            residual_in_drug = attn_out + conv_xd
            drug_features_attn = self.layer_norm_drug(residual_in_drug)
            drug_concat = torch.cat([drug_features_attn, fingerprint_features], dim=1)
            drug_features = self.fusion_drug_final(drug_concat)

        # ============= miRNA Processing (保持不变) =============
        embedded_xt = self.embedding_xt(target).permute(0, 2, 1)

        conv_xt1 = self.conv_xt_11(embedded_xt)
        conv_xt1 = self.relu(conv_xt1)
        conv_xt1 = self.dropout(conv_xt1)
        conv_xt1 = self.conv_xt_12(conv_xt1)
        conv_xt1 = self.relu(conv_xt1)
        conv_xt1 = F.max_pool1d(conv_xt1, conv_xt1.size(2)).squeeze(2)

        conv_xt2 = self.conv_xt_21(embedded_xt)
        conv_xt2 = self.relu(conv_xt2)
        conv_xt2 = self.dropout(conv_xt2)
        conv_xt2 = self.conv_xt_22(conv_xt2)
        conv_xt2 = self.relu(conv_xt2)
        conv_xt2 = F.max_pool1d(conv_xt2, conv_xt2.size(2)).squeeze(2)

        conv_xt3 = self.conv_xt_31(embedded_xt)
        conv_xt3 = self.relu(conv_xt3)
        conv_xt3 = self.dropout(conv_xt3)
        conv_xt3 = F.max_pool1d(conv_xt3, kernel_size=2)
        conv_xt3 = self.conv_xt_32(conv_xt3)
        conv_xt3 = self.relu(conv_xt3)
        conv_xt3 = F.max_pool1d(conv_xt3, conv_xt3.size(2)).squeeze(2)

        conv_xt = torch.cat((conv_xt1, conv_xt2, conv_xt3), dim=1).unsqueeze(2)
        conv_xt = self.conv_reduce_xt(conv_xt).squeeze(2)
        conv_xt = torch.nan_to_num(conv_xt, nan=0.0)
        mirna_seq_features = conv_xt

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

        # ============ 【改进点2】miRNA 特征融合 ============
        if self.use_improved_fusion:
            # 使用双向注意力融合
            mirna_features = self.mirna_fusion(mirna_seq_features, mirna_cgr_features)
        else:
            # 原版单向注意力
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

        # ============= CCL-ASPS 对比损失计算 =============
        if return_contrastive_loss:
            mirna_seq_norm = F.normalize(mirna_seq_features, dim=1)
            mirna_cgr_norm = F.normalize(mirna_cgr_features, dim=1)
            mirna_fused_norm = F.normalize(mirna_features, dim=1)

            drug_seq_norm = F.normalize(drug_seq_features, dim=1)
            drug_mol_norm = F.normalize(drug_mol_features, dim=1)
            drug_fused_norm = F.normalize(drug_features, dim=1)

            args_sim = {
                'current_epoch': current_epoch,
                'epochs': total_epochs,
                'beta': 0.8,
                'warmup_epochs': warmup_epochs
            }

            mirna_sim_matrix = torch.mm(mirna_fused_norm, mirna_fused_norm.t())
            pos_mask, neg_mask = get_contrast_pair_batch(args_sim, mirna_sim_matrix, data.target.device)
            loss_mirna_seq = self.contrast_mirna(mirna_seq_norm, mirna_fused_norm, pos_mask, neg_mask)
            loss_mirna_cgr = self.contrast_mirna(mirna_cgr_norm, mirna_fused_norm, pos_mask, neg_mask)
            loss_mirna_contrastive = loss_mirna_seq + loss_mirna_cgr

            drug_sim_matrix = torch.mm(drug_fused_norm, drug_fused_norm.t())
            pos_mask_d, neg_mask_d = get_contrast_pair_batch(args_sim, drug_sim_matrix, data.target.device)
            loss_drug_seq = self.contrast_drug(drug_seq_norm, drug_fused_norm, pos_mask_d, neg_mask_d)
            loss_drug_mol = self.contrast_drug(drug_mol_norm, drug_fused_norm, pos_mask_d, neg_mask_d)
            loss_drug_contrastive = loss_drug_seq + loss_drug_mol

            loss_dict = {
                'contrastive_mirna': loss_mirna_contrastive,
                'contrastive_drug': loss_drug_contrastive,
            }

            return out, loss_dict

        return out


# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    # 创建模型 (use_improved_fusion=True 启用改进)
    model_improved = AttnFusionGCNNet(use_improved_fusion=True)
    model_original = AttnFusionGCNNet(use_improved_fusion=False)

    print(f"改进版模型参数量: {sum(p.numel() for p in model_improved.parameters()):,}")
    print(f"原版模型参数量: {sum(p.numel() for p in model_original.parameters()):,}")
    print("\n使用方式:")
    print("model = AttnFusionGCNNet(use_improved_fusion=True)  # 启用双向注意力")
    print("model = AttnFusionGCNNet(use_improved_fusion=False) # 使用原版单向注意力")