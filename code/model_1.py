import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp
import warnings


class AttnFusionGCNNet(torch.nn.Module):
    """
    这是一个精简版的模型，只实现了 'attn_fusion' 逻辑，
    移除了所有其他的 ablation_mode 分支。
    """

    def __init__(self, n_output=1, n_filters=32, embed_dim=64, num_features_xd=78,
                 num_features_smile=66, num_features_xt=25, output_dim=128, dropout=0.2):
        super(AttnFusionGCNNet, self).__init__()

        # 警告：ablation_mode 已被移除。这个类只实现 attn_fusion。

        self.n_output = n_output
        self.output_dim = output_dim

        # --- [!!! 回退点: 恢复 "方案 B" 索引 !!!] ---
        self.max_smile_idx = num_features_smile
        self.max_target_idx = num_features_xt

        # --- [!!! 回退点: 恢复 "方案 B" Embedding 大小 !!!] ---
        self.smile_embed = nn.Embedding(num_features_smile + 1, embed_dim)

        self.conv_xd_11 = nn.Conv1d(embed_dim, out_channels=n_filters, kernel_size=3, padding=1)
        self.conv_xd_12 = nn.Conv1d(n_filters, out_channels=n_filters * 2, kernel_size=3, padding=1)
        self.conv_xd_21 = nn.Conv1d(embed_dim, out_channels=n_filters, kernel_size=2, padding=1)
        self.conv_xd_22 = nn.Conv1d(n_filters, out_channels=n_filters * 2, kernel_size=2, padding=1)
        self.conv_xd_31 = nn.Conv1d(embed_dim, out_channels=n_filters, kernel_size=1, padding=1)
        self.conv_xd_32 = nn.Conv1d(n_filters, out_channels=n_filters * 2, kernel_size=1, padding=1)
        self.fc_smiles = nn.Linear(n_filters * 2, output_dim)

        # ... (指纹特征定义 - 不变) ...
        self.rdkit_descriptor_dim = 210
        self.rdkit_fingerprint_dim = 136
        self.maccs_dim = 166
        self.morgan_dim = 512
        self.combined_dim = 1024
        self.attention_rdkit_descriptor = nn.Linear(self.rdkit_descriptor_dim, self.rdkit_descriptor_dim)
        self.attention_maccs = nn.Linear(self.maccs_dim, self.maccs_dim)
        self.drug_fingerprint_transform = nn.Linear(self.combined_dim, output_dim)

        # Drug Feature Fusion with Attention
        self.drug_attn = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )

        # --- [保留 LayerNorm 修复] ---
        self.layer_norm_drug = nn.LayerNorm(output_dim, eps=1e-3)
        # =================================

        # --- [保留 LeakyReLU 修复] ---
        self.relu = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(dropout)

        self.fusion_drug_final = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            self.relu,  # 使用 LeakyReLU
            self.dropout
        )
        # ========================================

        # ... (Conv reduce - 不变) ...
        self.conv_reduce_smiles = nn.Conv1d(in_channels=output_dim * 3, out_channels=output_dim, kernel_size=1)
        self.conv_reduce_xt = nn.Conv1d(in_channels=192, out_channels=output_dim, kernel_size=1)

        # --- [!!! 回退点: 恢复 "方案 B" Embedding 大小 !!!] ---
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_11 = nn.Conv1d(embed_dim, out_channels=n_filters, kernel_size=4, padding=2)
        self.conv_xt_12 = nn.Conv1d(n_filters, out_channels=n_filters * 2, kernel_size=4, padding=2)
        self.conv_xt_21 = nn.Conv1d(embed_dim, out_channels=n_filters, kernel_size=3, padding=1)
        self.conv_xt_22 = nn.Conv1d(n_filters, out_channels=n_filters * 2, kernel_size=3, padding=1)
        self.conv_xt_31 = nn.Conv1d(embed_dim, out_channels=n_filters, kernel_size=2, padding=1)
        self.conv_xt_32 = nn.Conv1d(n_filters, out_channels=n_filters * 2, kernel_size=2, padding=1)

        # miRNA structure matrix branch (硬编码 - 之前 'baseline', 'attn_fusion', 'no_sequence' 需要)
        self.conv_matrix_1 = nn.Conv2d(1, n_filters, kernel_size=3, padding=1)
        self.conv_matrix_2 = nn.Conv2d(n_filters, n_filters * 2, kernel_size=3, padding=1)
        self.conv_matrix_3 = nn.Conv2d(n_filters * 2, n_filters * 4, kernel_size=3, padding=1)
        self.fc_matrix_1 = nn.Linear(n_filters * 4 * 4 * 4, 256)
        self.fc_matrix_2 = nn.Linear(256, output_dim)

        # Attention fusion for miRNA (硬编码 - 之前 'attn_fusion' 需要)
        self.mirna_attn = nn.MultiheadAttention(embed_dim=output_dim, num_heads=8, batch_first=True, dropout=0.05)
        # --- [保留 LayerNorm 修复] ---
        self.layer_norm_mirna = nn.LayerNorm(output_dim, eps=1e-3)
        # =================================

        # (移除了 'baseline' 的 self.fusion_mirna)

        # Combined layers (移除 Sigmoid)
        self.fc1 = nn.Linear(output_dim * 2, 256)
        self.out = nn.Linear(256, self.n_output)

        # --- [!!! 回退点: 恢复 Sigmoid (因为 train.py 使用 BCELoss) !!!] ---
        self.ac = nn.Sigmoid()

    def process_drug_fingerprints(self, rdkit_descriptor, rdkit_fingerprint, maccs_fingerprint, morgan_fingerprint):
        """Process drug fingerprint features with self-attention"""
        # (Handle 1D/3D tensors and Validate dimensions - 不变)
        if len(rdkit_descriptor.shape) == 1: rdkit_descriptor = rdkit_descriptor.unsqueeze(0)
        if len(rdkit_fingerprint.shape) == 1: rdkit_fingerprint = rdkit_fingerprint.unsqueeze(0)
        if len(maccs_fingerprint.shape) == 1: maccs_fingerprint = maccs_fingerprint.unsqueeze(0)
        if len(morgan_fingerprint.shape) == 1: morgan_fingerprint = morgan_fingerprint.unsqueeze(0)
        with warnings.catch_warnings():  # 抑制 .mean() 对 2D 张量的警告
            warnings.simplefilter("ignore", category=UserWarning)
            if len(rdkit_descriptor.shape) > 2: rdkit_descriptor = rdkit_descriptor.mean(dim=1)
            if len(rdkit_fingerprint.shape) > 2: rdkit_fingerprint = rdkit_fingerprint.mean(dim=1)
            if len(maccs_fingerprint.shape) > 2: maccs_fingerprint = maccs_fingerprint.mean(dim=1)
            if len(morgan_fingerprint.shape) > 2: morgan_fingerprint = morgan_fingerprint.mean(dim=1)

        assert rdkit_descriptor.shape[
                   -1] == self.rdkit_descriptor_dim, f"rdkit_descriptor dim mismatch. Expected {self.rdkit_descriptor_dim}, got {rdkit_descriptor.shape[-1]}"
        assert rdkit_fingerprint.shape[-1] == self.rdkit_fingerprint_dim, "rdkit_fingerprint dim mismatch"
        assert maccs_fingerprint.shape[-1] == self.maccs_dim, "maccs_fingerprint dim mismatch"
        assert morgan_fingerprint.shape[-1] == self.morgan_dim, "morgan_fingerprint dim mismatch"

        # Apply self-attention
        attention_weights_rdkit = self.attention_rdkit_descriptor(rdkit_descriptor)
        attention_weights_rdkit = F.softmax(attention_weights_rdkit, dim=-1)
        rdkit_descriptor_prime = rdkit_descriptor * attention_weights_rdkit

        attention_weights_maccs = self.attention_maccs(maccs_fingerprint)
        attention_weights_maccs = F.softmax(attention_weights_maccs, dim=-1)
        maccs_prime = maccs_fingerprint * attention_weights_maccs

        # Concatenate
        combined_features = torch.cat([
            rdkit_descriptor_prime,
            maccs_prime,
            rdkit_fingerprint,
            morgan_fingerprint
        ], dim=-1)

        # Transform
        drug_features = self.drug_fingerprint_transform(combined_features)
        drug_features = self.relu(drug_features)  # <-- LeakyReLU
        drug_features = self.dropout(drug_features)

        # --- [!!! 关键: 保留 nan/inf 清理 !!!] ---
        drug_features = torch.nan_to_num(drug_features, nan=0.0, posinf=0.0, neginf=0.0)
        return drug_features

    def forward(self, data):
        # ============= Step 1: 获取数据并检查 =============
        rdkit_fingerprint = data.rdkit_fingerprint
        rdkit_descriptor = data.rdkit_descriptor
        maccs_fingerprint = data.maccs_fingerprint
        morgan_fingerprint = data.morgan_fingerprint
        drugsmile = data.seqdrug
        target = data.target
        target_matrix = data.target_matrix if hasattr(data, 'target_matrix') else None

        if target_matrix is None:
            raise ValueError("target_matrix is None. 'attn_fusion' logic requires target_matrix.")

        # ============= Step 2: 数据类型和边界检查 =============
        if drugsmile.dtype == torch.float32 or drugsmile.dtype == torch.float64: drugsmile = drugsmile.long()
        if target.dtype == torch.float32 or target.dtype == torch.float64: target = target.long()

        drugsmile_max = drugsmile.max().item()
        drugsmile_min = drugsmile.min().item()
        target_max = target.max().item()
        target_min = target.min().item()

        # (方案 B: self.max_smile_idx 现在是 66, 这个检查会通过)
        if drugsmile_max > self.max_smile_idx: drugsmile = torch.clamp(drugsmile, 0, self.max_smile_idx)
        if drugsmile_min < 0: drugsmile = torch.clamp(drugsmile, 0, self.max_smile_idx)
        if target_max > self.max_target_idx: target = torch.clamp(target, 0, self.max_target_idx)
        if target_min < 0: target = torch.clamp(target, 0, self.max_target_idx)

        # ============= Step 3: 计算 batch_size =============
        batch_size = drugsmile.shape[0]

        # ============= Step 4: Reshape 指纹特征 =============
        try:
            rdkit_descriptor = rdkit_descriptor.view(batch_size, self.rdkit_descriptor_dim)
            rdkit_fingerprint = rdkit_fingerprint.view(batch_size, self.rdkit_fingerprint_dim)
            maccs_fingerprint = maccs_fingerprint.view(batch_size, self.maccs_dim)
            morgan_fingerprint = morgan_fingerprint.view(batch_size, self.morgan_dim)
        except RuntimeError as e:
            print(f"[ERROR] Reshape failed for fingerprints. Batch size: {batch_size}")
            raise e

        # --- [!!! 关键: 保留 nan/inf 清理 !!!] ---
        rdkit_descriptor = torch.nan_to_num(rdkit_descriptor, nan=0.0, posinf=0.0, neginf=0.0)
        rdkit_fingerprint = torch.nan_to_num(rdkit_fingerprint, nan=0.0, posinf=0.0, neginf=0.0)
        maccs_fingerprint = torch.nan_to_num(maccs_fingerprint, nan=0.0, posinf=0.0, neginf=0.0)
        morgan_fingerprint = torch.nan_to_num(morgan_fingerprint, nan=0.0, posinf=0.0, neginf=0.0)
        target_matrix = torch.nan_to_num(target_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        # ===============================================

        # ============= Drug Processing =============
        # 1. Process drug fingerprints
        fingerprint_features = self.process_drug_fingerprints(
            rdkit_descriptor=rdkit_descriptor,
            rdkit_fingerprint=rdkit_fingerprint,
            maccs_fingerprint=maccs_fingerprint,
            morgan_fingerprint=morgan_fingerprint
        )  # (已清理)

        # 2. SMILES sequence processing
        try:
            embedded_smile = self.smile_embed(drugsmile)
        except RuntimeError as e:
            print(f"[ERROR] smile_embed failed")
            raise e
        embedded_smile = embedded_smile.permute(0, 2, 1)

        # --- (保留 LeakyReLU 和 dropout 修复) ---
        conv_xd1 = self.conv_xd_11(embedded_smile)
        conv_xd1 = self.relu(conv_xd1)
        conv_xd1 = self.dropout(conv_xd1)
        conv_xd1 = F.max_pool1d(conv_xd1, kernel_size=2)
        conv_xd1 = self.conv_xd_12(conv_xd1)
        conv_xd1 = self.relu(conv_xd1)
        conv_xd1 = F.max_pool1d(conv_xd1, conv_xd1.size(2)).squeeze(2)

        conv_xd2 = self.conv_xd_21(embedded_smile)
        conv_xd2 = self.relu(conv_xd2)
        conv_xd2 = self.dropout(conv_xd2)
        conv_xd2 = F.max_pool1d(conv_xd2, kernel_size=2)
        conv_xd2 = self.conv_xd_22(conv_xd2)
        conv_xd2 = self.relu(conv_xd2)
        conv_xd2 = self.dropout(conv_xd2)
        conv_xd2 = F.max_pool1d(conv_xd2, conv_xd2.size(2)).squeeze(2)

        conv_xd3 = self.conv_xd_31(embedded_smile)
        conv_xd3 = self.relu(conv_xd3)
        conv_xd3 = self.dropout(conv_xd3)
        conv_xd3 = F.max_pool1d(conv_xd3, kernel_size=2)
        conv_xd3 = self.conv_xd_32(conv_xd3)
        conv_xd3 = self.relu(conv_xd3)
        conv_xd3 = F.max_pool1d(conv_xd3, conv_xd3.size(2)).squeeze(2)
        # --- [修复结束] ---

        conv_xd1 = self.fc_smiles(conv_xd1)
        conv_xd2 = self.fc_smiles(conv_xd2)
        conv_xd3 = self.fc_smiles(conv_xd3)

        conv_xd = torch.cat((conv_xd1, conv_xd2, conv_xd3), dim=1)
        conv_xd = conv_xd.unsqueeze(1).permute(0, 2, 1)
        conv_xd = self.conv_reduce_smiles(conv_xd)
        conv_xd = conv_xd.squeeze(2)

        # --- [!!! 关键: 保留 nan/inf 清理 !!!] ---
        conv_xd = torch.nan_to_num(conv_xd, nan=0.0, posinf=0.0, neginf=0.0)

        # 3. Drug Feature Fusion with Attention
        smiles_unsq = conv_xd.unsqueeze(1)
        fingerprint_unsq = fingerprint_features.unsqueeze(1)

        attn_out, attn_weights = self.drug_attn(
            query=smiles_unsq,
            key=fingerprint_unsq,
            value=fingerprint_unsq
        )
        attn_out = attn_out.squeeze(1)

        # --- [!!! 关键: 保留 nan/inf 清理 (在 LayerNorm 之前) !!!] ---
        attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=0.0, neginf=0.0)  # 1. 清理 MHA
        residual_in_drug = attn_out + conv_xd  # 2. 创建残差
        residual_in_drug = torch.nan_to_num(residual_in_drug, nan=0.0, posinf=0.0, neginf=0.0)  # 3. 清理残差

        drug_features_attn = self.layer_norm_drug(residual_in_drug)  # 4. LayerNorm (eps=1e-3)
        drug_features_attn = torch.nan_to_num(drug_features_attn, nan=0.0, posinf=0.0,
                                              neginf=0.0)  # 5. 清理 LayerNorm 的输出
        # =================================================

        drug_concat = torch.cat([drug_features_attn, fingerprint_features], dim=1)
        drug_features = self.fusion_drug_final(drug_concat)  # 内部使用 LeakyReLU

        # ============= miRNA Sequence Processing =============
        # (硬编码 - 之前 'no_sequence' 会跳过)
        try:
            embedded_xt = self.embedding_xt(target)
        except RuntimeError as e:
            print(f"[ERROR] embedding_xt failed")
            raise e
        embedded_xt = embedded_xt.permute(0, 2, 1)

        # --- (保留 LeakyReLU 和 dropout 修复) ---
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
        # --- [修复结束] ---

        conv_xt = torch.cat((conv_xt1, conv_xt2, conv_xt3), dim=1)
        conv_xt = conv_xt.unsqueeze(2)
        conv_xt = self.conv_reduce_xt(conv_xt)
        conv_xt = conv_xt.squeeze(2)

        # --- [!!! 关键: 保留 nan/inf 清理 !!!] ---
        conv_xt = torch.nan_to_num(conv_xt, nan=0.0, posinf=0.0, neginf=0.0)

        # ============= miRNA Matrix Processing =============
        # (硬编码 - 之前 'baseline', 'attn_fusion', 'no_sequence' 会执行)
        if len(target_matrix.shape) == 3: target_matrix = target_matrix.unsqueeze(1)

        matrix_feat = F.max_pool2d(self.relu(self.conv_matrix_1(target_matrix)), kernel_size=2)  # LeakyReLU
        matrix_feat = F.max_pool2d(self.relu(self.conv_matrix_2(matrix_feat)), kernel_size=2)  # LeakyReLU
        matrix_feat = self.dropout(self.relu(self.conv_matrix_3(matrix_feat)))  # LeakyReLU

        matrix_feat = matrix_feat.view(matrix_feat.size(0), -1)
        matrix_feat = self.dropout(self.relu(self.fc_matrix_1(matrix_feat)))  # LeakyReLU
        matrix_feat = self.fc_matrix_2(matrix_feat)

        # --- [!!! 关键: 保留 nan/inf 清理 !!!] ---
        matrix_feat = torch.nan_to_num(matrix_feat, nan=0.0, posinf=0.0, neginf=0.0)

        # ============= miRNA Fusion =============
        # (硬编码 - 只使用 'attn_fusion' 逻辑)
        xt_unsq = conv_xt.unsqueeze(1)
        mat_unsq = matrix_feat.unsqueeze(1)
        attn_out, _ = self.mirna_attn(query=xt_unsq, key=mat_unsq, value=mat_unsq)
        attn_out = attn_out.squeeze(1)

        # --- [!!! 关键: 保留 nan/inf 清理 (在 LayerNorm 之前) !!!] ---
        attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=0.0, neginf=0.0)  # 1. 清理 MHA
        residual_in_mirna = attn_out + conv_xt  # 2. 创建残差
        residual_in_mirna = torch.nan_to_num(residual_in_mirna, nan=0.0, posinf=0.0, neginf=0.0)  # 3. 清理残差

        mirna_features = self.layer_norm_mirna(residual_in_mirna)  # 4. LayerNorm (eps=1e-3)
        mirna_features = torch.nan_to_num(mirna_features, nan=0.0, posinf=0.0, neginf=0.0)  # 5. 清理 LayerNorm 的输出
        # =================================================

        mirna_features = self.relu(mirna_features)  # LeakyReLU

        # (移除了 'baseline' 和其他分支的融合逻辑)

        # ============= Final Combination =============
        xc = torch.cat((drug_features, mirna_features), dim=1)
        xc = self.fc1(xc)
        xc = self.relu(xc)  # LeakyReLU
        xc = self.dropout(xc)
        out = self.out(xc)

        # --- [!!! 回退点: 恢复 Sigmoid !!!] ---
        out = self.ac(out)

        return out