import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

# ==========================================
# 主模型 (AttnFusionGCNNet) - 无对比学习版 (Ablation)
# ==========================================

class AttnFusionGCNNet(torch.nn.Module):
    def __init__(self, n_output=1, n_filters=32, embed_dim=64, num_features_xd=78,
                 num_features_smile=66, num_features_xt=25, output_dim=128, dropout=0.2):
        """
        移除 contrastive_dim, temperature, lam 等对比学习参数
        """
        super(AttnFusionGCNNet, self).__init__()

        self.n_output = n_output
        self.output_dim = output_dim
        
        # Embedding 参数
        self.max_smile_idx = num_features_smile
        self.max_target_idx = num_features_xt
        self.smile_embed = nn.Embedding(num_features_smile + 1, embed_dim)

        # ============ Drug Encoders (保持不变) ============
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

        # 1. miRNA Sequence CNNs (保持不变)
        self.conv_xt_11 = nn.Conv1d(embed_dim, out_channels=n_filters, kernel_size=4, padding=2)
        self.conv_xt_12 = nn.Conv1d(n_filters, out_channels=n_filters * 2, kernel_size=4, padding=2)
        self.conv_xt_21 = nn.Conv1d(embed_dim, out_channels=n_filters, kernel_size=3, padding=1)
        self.conv_xt_22 = nn.Conv1d(n_filters, out_channels=n_filters * 2, kernel_size=3, padding=1)
        self.conv_xt_31 = nn.Conv1d(embed_dim, out_channels=n_filters, kernel_size=2, padding=1)
        self.conv_xt_32 = nn.Conv1d(n_filters, out_channels=n_filters * 2, kernel_size=2, padding=1)

        # 2. miRNA Matrix CNNs
        # 输入: [Batch, 1, 16, 16]
        self.conv_matrix_1 = nn.Conv2d(1, n_filters, kernel_size=3, padding=1)
        self.conv_matrix_2 = nn.Conv2d(n_filters, n_filters * 2, kernel_size=3, padding=1)
        self.conv_matrix_3 = nn.Conv2d(n_filters * 2, n_filters * 4, kernel_size=3, padding=1)
        
        self.flatten_dim = (n_filters * 4) * 4 * 4
        
        self.fc_matrix_1 = nn.Linear(self.flatten_dim, 256)
        self.fc_matrix_2 = nn.Linear(256, output_dim)

        # miRNA Attention
        self.mirna_attn = nn.MultiheadAttention(embed_dim=output_dim, num_heads=8, batch_first=True, dropout=0.05)
        self.layer_norm_mirna = nn.LayerNorm(output_dim, eps=1e-3)

        # miRNA 最终融合层
        self.fusion_mirna_final = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            self.relu,
            self.dropout
        )

        # ============ 移除对比学习模块 ============
        # self.contrast_drug = ... (Deleted)
        # self.contrast_mirna = ... (Deleted)

        # Final layers
        self.fc1 = nn.Linear(output_dim * 2, 256)
        self.out = nn.Linear(256, self.n_output)
        self.ac = nn.Sigmoid()

    def process_drug_fingerprints(self, rdkit_descriptor, rdkit_fingerprint, maccs_fingerprint, morgan_fingerprint):
        # 保持原有的指纹处理逻辑不变
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

    def forward(self, data):
        # 移除 contrastive related arguments: current_epoch, total_epochs, warmup_epochs, return_contrastive_loss
        
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
        # 1. Drug Fingerprint (View 1)
        fingerprint_features = self.process_drug_fingerprints(
            rdkit_descriptor, rdkit_fingerprint, maccs_fingerprint, morgan_fingerprint
        )
        
        # 2. SMILES Sequence (View 2)
        embedded_smile = self.smile_embed(drugsmile).permute(0, 2, 1)
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

        conv_xd1 = self.fc_smiles(conv_xd1)
        conv_xd2 = self.fc_smiles(conv_xd2)
        conv_xd3 = self.fc_smiles(conv_xd3)

        conv_xd = torch.cat((conv_xd1, conv_xd2, conv_xd3), dim=1).unsqueeze(1).permute(0, 2, 1)
        conv_xd = self.conv_reduce_smiles(conv_xd).squeeze(2)
        conv_xd = torch.nan_to_num(conv_xd, nan=0.0)

        # 3. Drug Fusion (保留 Attention)
        smiles_unsq = conv_xd.unsqueeze(1)
        fingerprint_unsq = fingerprint_features.unsqueeze(1)
        attn_out, _ = self.drug_attn(query=smiles_unsq, key=fingerprint_unsq, value=fingerprint_unsq)
        attn_out = torch.nan_to_num(attn_out.squeeze(1), nan=0.0)
        residual_in_drug = attn_out + conv_xd
        drug_features_attn = self.layer_norm_drug(residual_in_drug)
        drug_concat = torch.cat([drug_features_attn, fingerprint_features], dim=1)
        drug_features = self.fusion_drug_final(drug_concat)

        # ============= miRNA Processing =============

        # 1. miRNA Sequence (View 1)
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

        # 2. miRNA Matrix (View 2)
        if len(target_matrix.shape) == 3: target_matrix = target_matrix.unsqueeze(1)
        
        matrix_feat = self.conv_matrix_1(target_matrix)
        matrix_feat = self.relu(matrix_feat)
        matrix_feat = F.max_pool2d(matrix_feat, kernel_size=2)
        
        matrix_feat = self.conv_matrix_2(matrix_feat)
        matrix_feat = self.relu(matrix_feat)
        matrix_feat = F.max_pool2d(matrix_feat, kernel_size=2)
        
        matrix_feat = self.conv_matrix_3(matrix_feat)
        matrix_feat = self.relu(matrix_feat)
        matrix_feat = self.dropout(matrix_feat)
        
        matrix_feat = matrix_feat.view(matrix_feat.size(0), -1)
        matrix_feat = self.fc_matrix_1(matrix_feat)
        matrix_feat = self.relu(matrix_feat)
        matrix_feat = self.dropout(matrix_feat)
        matrix_feat = self.fc_matrix_2(matrix_feat)
        matrix_feat = torch.nan_to_num(matrix_feat, nan=0.0)

        # 3. miRNA Fusion (Fused View) (保留 Attention)
        xt_unsq = conv_xt.unsqueeze(1)
        mat_unsq = matrix_feat.unsqueeze(1)

        attn_out_m, _ = self.mirna_attn(query=xt_unsq, key=mat_unsq, value=mat_unsq)
        attn_out_m = torch.nan_to_num(attn_out_m.squeeze(1), nan=0.0)

        residual_in_mirna = attn_out_m + conv_xt
        mirna_features_attn = self.layer_norm_mirna(residual_in_mirna)
        
        mirna_concat = torch.cat([mirna_features_attn, matrix_feat], dim=1)
        mirna_features = self.fusion_mirna_final(mirna_concat)

        # ============= Final Prediction =============
        xc = torch.cat((drug_features, mirna_features), dim=1)
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        out = self.ac(out)

        # ============= 移除 Return Contrastive Loss 逻辑 =============
        # 直接返回预测值
        return out