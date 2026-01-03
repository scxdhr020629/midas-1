import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings


# ==========================================
# 1. CCL 核心对比模块 (Model_Contrast)
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

        # 先计算余弦相似度，再除以温度，最后 exp
        cos_sim = dot_numerator / (dot_denominator + 1e-8)
        sim_matrix = torch.exp(cos_sim / self.tau)
        return sim_matrix

    def forward(self, v1_embs, v2_embs, pos=None, neg=None):
        """
        InfoNCE 损失计算
        """
        v1_embs = self.proj(v1_embs)
        v2_embs = self.proj(v2_embs)

        # 计算相似度矩阵
        matrix_1to2 = self.sim(v1_embs, v2_embs)

        # 应用掩码: 正样本和负样本的相似度
        pos_sim = (matrix_1to2 * pos).sum(dim=1)  # [batch_size]
        neg_sim = (matrix_1to2 * neg).sum(dim=1)  # [batch_size]

        # InfoNCE Loss
        loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8) + 1e-8)

        return loss.mean()


# ==========================================
# 2. 采样策略
# ==========================================
def get_contrast_pair_batch(batch_size, device):
    """
    标准对比学习采样策略：
    - 正样本 (Positive): 对角线 (自身与其他视图的自身)
    - 负样本 (Negative): 除对角线外的所有样本
    """
    # 1. 基础正样本 (Positive): 对角线
    pos = torch.eye(batch_size).to(device)

    # 2. 基础负样本 (Negative): 所有非对角线
    neg = torch.ones(batch_size, batch_size).to(device) - pos

    return pos, neg


# ==========================================
# 3. 消融实验模型 (AttnFusionGCNNet_Ablation)
# ==========================================
class AttnFusionGCNNet_Ablation(torch.nn.Module):
    """
    支持以下消融模式:
    - 'full': 完整模型
    - 'no_mirna_seq': 消融miRNA序列特征 (m1)
    - 'no_mirna_cgr': 消融miRNA CGR特征 (m2)
    - 'no_drug_seq': 消融drug序列特征 (d1)
    - 'no_drug_fp': 消融drug指纹特征 (d2)
    - 'no_attention': 消融交叉注意力
    - 'no_contrastive': 消融协同对比学习
    - 'no_mirna_seq_drug_seq': 同时消融miRNA序列和drug序列 (m1+d1)
    - 'no_mirna_cgr_drug_fp': 同时消融miRNA CGR和drug指纹 (m2+d2)
    """
    def __init__(self, ablation_mode='full', n_output=1, n_filters=32, embed_dim=64, 
                 num_features_xd=78, num_features_smile=66, num_features_xt=25, 
                 output_dim=128, dropout=0.2, contrastive_dim=128, temperature=0.1, lam=0.5):
        super(AttnFusionGCNNet_Ablation, self).__init__()

        self.ablation_mode = ablation_mode
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

        # 1. miRNA Sequence CNNs
        self.conv_xt_11 = nn.Conv1d(embed_dim, out_channels=n_filters, kernel_size=4, padding=2)
        self.conv_xt_12 = nn.Conv1d(n_filters, out_channels=n_filters * 2, kernel_size=4, padding=2)
        self.conv_xt_21 = nn.Conv1d(embed_dim, out_channels=n_filters, kernel_size=3, padding=1)
        self.conv_xt_22 = nn.Conv1d(n_filters, out_channels=n_filters * 2, kernel_size=3, padding=1)
        self.conv_xt_31 = nn.Conv1d(embed_dim, out_channels=n_filters, kernel_size=2, padding=1)
        self.conv_xt_32 = nn.Conv1d(n_filters, out_channels=n_filters * 2, kernel_size=2, padding=1)

        # 2. miRNA Matrix CNNs
        self.conv_matrix_1 = nn.Conv2d(1, n_filters, kernel_size=3, padding=1)
        self.conv_matrix_2 = nn.Conv2d(n_filters, n_filters * 2, kernel_size=3, padding=1)
        self.conv_matrix_3 = nn.Conv2d(n_filters * 2, n_filters * 4, kernel_size=3, padding=1)
        
        self.flatten_dim = (n_filters * 4) * 4 * 4
        
        self.fc_matrix_1 = nn.Linear(self.flatten_dim, 256)
        self.fc_matrix_2 = nn.Linear(256, output_dim)

        # miRNA Attention
        self.mirna_attn = nn.MultiheadAttention(embed_dim=output_dim, num_heads=8, batch_first=True, dropout=0.05)
        self.layer_norm_mirna = nn.LayerNorm(output_dim, eps=1e-3)

        self.fusion_mirna_final = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            self.relu,
            self.dropout
        )

        # ============ 对比学习模块 ============
        self.contrast_drug = Model_Contrast(hidden_dim=output_dim, tau=temperature, lam=lam)
        self.contrast_mirna = Model_Contrast(hidden_dim=output_dim, tau=temperature, lam=lam)

        # Final layers
        self.fc1 = nn.Linear(output_dim * 2, 256)
        self.out = nn.Linear(256, self.n_output)
        self.ac = nn.Sigmoid()

    def process_drug_fingerprints(self, rdkit_descriptor, rdkit_fingerprint, maccs_fingerprint, morgan_fingerprint):
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

        rdkit_descriptor = torch.nan_to_num(rdkit_descriptor, nan=0.0)
        rdkit_fingerprint = torch.nan_to_num(rdkit_fingerprint, nan=0.0)
        maccs_fingerprint = torch.nan_to_num(maccs_fingerprint, nan=0.0)
        morgan_fingerprint = torch.nan_to_num(morgan_fingerprint, nan=0.0)
        target_matrix = torch.nan_to_num(target_matrix, nan=0.0)

        # ============= 判断消融模式 =============
        # 新增组合消融模式的判断
        use_drug_fp = self.ablation_mode not in ['no_drug_fp', 'no_mirna_cgr_drug_fp']
        use_drug_seq = self.ablation_mode not in ['no_drug_seq', 'no_mirna_seq_drug_seq']
        use_mirna_seq = self.ablation_mode not in ['no_mirna_seq', 'no_mirna_seq_drug_seq']
        use_mirna_cgr = self.ablation_mode not in ['no_mirna_cgr', 'no_mirna_cgr_drug_fp']
        use_attention = self.ablation_mode != 'no_attention'

        # ============= Drug Processing =============
        # Drug Fingerprint Features (d2)
        if use_drug_fp:
            fingerprint_features = self.process_drug_fingerprints(
                rdkit_descriptor, rdkit_fingerprint, maccs_fingerprint, morgan_fingerprint
            )
            drug_mol_features = fingerprint_features
        else:
            drug_mol_features = None

        # Drug Sequence Features (d1)
        if use_drug_seq:
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
            drug_seq_features = conv_xd
        else:
            drug_seq_features = None

        # Drug Feature Fusion
        if use_drug_seq and use_drug_fp:
            if use_attention:
                # 使用交叉注意力融合
                smiles_unsq = conv_xd.unsqueeze(1)
                fingerprint_unsq = fingerprint_features.unsqueeze(1)
                attn_out, _ = self.drug_attn(query=smiles_unsq, key=fingerprint_unsq, value=fingerprint_unsq)
                attn_out = torch.nan_to_num(attn_out.squeeze(1), nan=0.0)
                residual_in_drug = attn_out + conv_xd
                drug_features_attn = self.layer_norm_drug(residual_in_drug)
                drug_concat = torch.cat([drug_features_attn, fingerprint_features], dim=1)
                drug_features = self.fusion_drug_final(drug_concat)
            else:
                # 不使用注意力，直接拼接
                drug_concat = torch.cat([conv_xd, fingerprint_features], dim=1)
                drug_features = self.fusion_drug_final(drug_concat)
        elif use_drug_seq:
            # 只有序列特征
            drug_features = conv_xd
        elif use_drug_fp:
            # 只有指纹特征
            drug_features = fingerprint_features
        else:
            raise ValueError("At least one drug feature must be enabled")

        # ============= miRNA Processing =============
        # miRNA Sequence Features (m1)
        if use_mirna_seq:
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
        else:
            mirna_seq_features = None

        # miRNA CGR Features (m2)
        if use_mirna_cgr:
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
            mirna_cgr_features = matrix_feat
        else:
            mirna_cgr_features = None

        # miRNA Feature Fusion
        if use_mirna_seq and use_mirna_cgr:
            if use_attention:
                # 使用交叉注意力融合
                xt_unsq = conv_xt.unsqueeze(1)
                mat_unsq = matrix_feat.unsqueeze(1)
                attn_out_m, _ = self.mirna_attn(query=xt_unsq, key=mat_unsq, value=mat_unsq)
                attn_out_m = torch.nan_to_num(attn_out_m.squeeze(1), nan=0.0)
                residual_in_mirna = attn_out_m + conv_xt
                mirna_features_attn = self.layer_norm_mirna(residual_in_mirna)
                mirna_concat = torch.cat([mirna_features_attn, matrix_feat], dim=1)
                mirna_features = self.fusion_mirna_final(mirna_concat)
            else:
                # 不使用注意力，直接拼接
                mirna_concat = torch.cat([conv_xt, matrix_feat], dim=1)
                mirna_features = self.fusion_mirna_final(mirna_concat)
        elif use_mirna_seq:
            # 只有序列特征
            mirna_features = conv_xt
        elif use_mirna_cgr:
            # 只有CGR特征
            mirna_features = matrix_feat
        else:
            raise ValueError("At least one miRNA feature must be enabled")

        # ============= Final Prediction =============
        xc = torch.cat((drug_features, mirna_features), dim=1)
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        out = self.ac(out)

        # ============= Contrastive Loss =============
        if return_contrastive_loss and self.ablation_mode != 'no_contrastive':
            # 准备对比学习特征
            mirna_seq_norm = F.normalize(mirna_seq_features, dim=1) if use_mirna_seq else None
            mirna_cgr_norm = F.normalize(mirna_cgr_features, dim=1) if use_mirna_cgr else None
            mirna_fused_norm = F.normalize(mirna_features, dim=1)

            drug_seq_norm = F.normalize(drug_seq_features, dim=1) if use_drug_seq else None
            drug_mol_norm = F.normalize(drug_mol_features, dim=1) if use_drug_fp else None
            drug_fused_norm = F.normalize(drug_features, dim=1)

            # 标准采样
            pos_mask, neg_mask = get_contrast_pair_batch(batch_size, data.target.device)

            # 计算对比损失
            loss_mirna_contrastive = 0.0
            if use_mirna_seq:
                loss_mirna_seq = self.contrast_mirna(mirna_seq_norm, mirna_fused_norm, pos_mask, neg_mask)
                loss_mirna_contrastive += loss_mirna_seq
            if use_mirna_cgr:
                loss_mirna_cgr = self.contrast_mirna(mirna_cgr_norm, mirna_fused_norm, pos_mask, neg_mask)
                loss_mirna_contrastive += loss_mirna_cgr

            loss_drug_contrastive = 0.0
            if use_drug_seq:
                loss_drug_seq = self.contrast_drug(drug_seq_norm, drug_fused_norm, pos_mask, neg_mask)
                loss_drug_contrastive += loss_drug_seq
            if use_drug_fp:
                loss_drug_mol = self.contrast_drug(drug_mol_norm, drug_fused_norm, pos_mask, neg_mask)
                loss_drug_contrastive += loss_drug_mol

            loss_dict = {
                'contrastive_mirna': loss_mirna_contrastive,
                'contrastive_drug': loss_drug_contrastive,
            }

            return out, loss_dict

        # 如果消融对比学习，返回空字典
        if return_contrastive_loss:
            loss_dict = {
                'contrastive_mirna': torch.tensor(0.0).to(out.device),
                'contrastive_drug': torch.tensor(0.0).to(out.device),
            }
            return out, loss_dict

        return out