import sys
import torch
import torch.nn as nn
import numpy as np
import itertools
from test_asps_model import AttnFusionGCNNet
from utils import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from torch_geometric.loader import DataLoader
import os
import time

# --- 全局配置 (保持不变的部分) ---
LOG_INTERVAL = 45
NUM_FOLDS = 5
loss_fn = nn.BCEWithLogitsLoss()

LR = 0.0005
WEIGHT_DECAY = 0.0032
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64

# --- 对比学习固定参数 (Loss 权重保持不变) ---
ALPHA = 0.5  # miRNA loss weight
BETA = 0.5   # Drug loss weight (全局固定，不参与网格搜索)
GAMMA = 1.0  # BCE loss weight
TEMPERATURE = 0.1
LAM = 0.5
CONTRASTIVE_DIM = 128

# ============================================================================
# 🔎 网格搜索参数空间 (ASPS 相关)
# ============================================================================
GRID_SEARCH_SPACE = {
    'NUM_EPOCHS': [20, 30, 40],           # 训练轮数
    'WARMUP_EPOCHS': [2,3,5,7],             # ASPS 预热轮数 (ASPS 参数)
    'ASPS_SAMPLING_RATE': [0.3,0.5, 0.7]      # ASPS 采样强度 (k_neg 系数, 模型内部的 beta)
}

# ============================================================================
# 辅助函数
# ============================================================================
def get_contrastive_weight(epoch, warmup_epochs):
    """计算对比学习权重的动态衰减/增长"""
    if epoch <= warmup_epochs:
        progress = epoch / warmup_epochs
        return 0.5 * (1 - np.cos(np.pi * progress))
    return 1.0

def train_epoch(model, device, train_loader, optimizer, epoch, total_epochs, warmup_epochs, asps_sampling_rate):
    model.train()
    total_loss = 0
    batch_count = 0
    contrastive_weight_factor = get_contrastive_weight(epoch, warmup_epochs)

    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)

        # 前向传播 - 【重要】传入搜索到的 ASPS 采样率参数
        output, loss_dict = model(
            data,
            current_epoch=epoch,
            total_epochs=total_epochs,
            warmup_epochs=warmup_epochs,
            asps_beta=asps_sampling_rate,  # 传入动态采样率
            return_contrastive_loss=True
        )

        labels = data.y.view(-1, 1).float().to(device)
        output = output.view(-1, 1)

        # 计算主损失
        loss_bce = loss_fn(output, labels)
        
        # 获取对比损失
        loss_mirna_contrastive = loss_dict['contrastive_mirna']
        loss_drug_contrastive = loss_dict['contrastive_drug']

        # 组合总损失 (使用全局固定的 ALPHA/BETA/GAMMA)
        loss = (GAMMA * loss_bce +
                contrastive_weight_factor * (ALPHA * loss_mirna_contrastive +
                                             BETA * loss_drug_contrastive))

        if torch.isnan(loss) or torch.isinf(loss):
            return None # 标记为失败

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

    return total_loss / batch_count

def evaluate(model, device, loader, total_epochs, warmup_epochs):
    model.eval()
    total_probs = []
    total_labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(
                data,
                current_epoch=0, 
                total_epochs=total_epochs,
                warmup_epochs=warmup_epochs,
                asps_beta=0.8, # 推理时不影响结果，给个默认值即可
                return_contrastive_loss=False
            )
            probs = torch.sigmoid(output).cpu().numpy().flatten()
            total_probs.extend(probs)
            total_labels.extend(data.y.view(-1, 1).cpu().numpy().flatten())

    total_probs = np.array(total_probs)
    total_labels = np.array(total_labels)
    
    try:
        roc_auc = roc_auc_score(total_labels, total_probs)
    except ValueError:
        roc_auc = 0.5

    return roc_auc

# ============================================================================
# 主网格搜索逻辑
# ============================================================================
def run_grid_search():
    cuda_name = "cuda:0"
    if len(sys.argv) > 1:
        cuda_name = "cuda:" + str(int(sys.argv[1]))
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}\n')

    # 生成所有参数组合
    keys, values = zip(*GRID_SEARCH_SPACE.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    best_overall_auc = 0.0
    best_params = None
    results_log = []

    print(f"开始网格搜索 (ASPS Tuning)，总共有 {len(param_combinations)} 种参数组合...\n")

    for idx, params in enumerate(param_combinations):
        current_epochs = params['NUM_EPOCHS']
        current_warmup = params['WARMUP_EPOCHS']
        current_asps_rate = params['ASPS_SAMPLING_RATE']
        
        print(f"{'='*30} Combination {idx+1}/{len(param_combinations)} {'='*30}")
        print(f"Params: Epochs={current_epochs}, Warmup={current_warmup}, ASPS_Rate={current_asps_rate}")

        fold_aucs = []

        # --- 5-Fold Cross Validation ---
        for fold in range(NUM_FOLDS):
            train_data = TestbedDataset(root='data', dataset='train' + str(fold))
            test_data = TestbedDataset(root='data', dataset='test' + str(fold))
            
            train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
            test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, drop_last=False)

            model = AttnFusionGCNNet(
                n_output=1, n_filters=32, embed_dim=64, num_features_xd=78,
                num_features_smile=66, num_features_xt=25, output_dim=128, dropout=0.2,
                contrastive_dim=CONTRASTIVE_DIM, temperature=TEMPERATURE, lam=LAM
            ).to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=current_epochs, eta_min=LR * 0.01)

            # Training Loop
            model_failed = False
            for epoch in range(1, current_epochs + 1):
                loss = train_epoch(model, device, train_loader, optimizer, epoch, 
                                 current_epochs, current_warmup, current_asps_rate)
                if loss is None:
                    print(f"  [Fold {fold}] Training failed (NaN loss). Skipping...")
                    model_failed = True
                    break
                scheduler.step()

            if model_failed:
                fold_aucs.append(0.0)
            else:
                # 使用最后一轮模型评估
                auc_score = evaluate(model, device, test_loader, current_epochs, current_warmup)
                fold_aucs.append(auc_score)
                print(f"  [Fold {fold}] Final AUC: {auc_score:.4f}")

        # 计算当前组合的平均性能
        mean_auc = np.mean(fold_aucs)
        print(f"--> Result: Mean AUC = {mean_auc:.4f}")
        
        results_log.append({
            'params': params,
            'auc': mean_auc
        })

        if mean_auc > best_overall_auc:
            best_overall_auc = mean_auc
            best_params = params
            print(f"🎉 New Best Found!")

    # ============================================================================
    # 最终结果输出
    # ============================================================================
    print("\n" + "="*70)
    print("🏆 GRID SEARCH COMPLETED")
    print("="*70)
    print(f"Best AUC: {best_overall_auc:.5f}")
    print(f"Best Parameters: {best_params}")
    print("-" * 70)
    print("All Results:")
    for res in sorted(results_log, key=lambda x: x['auc'], reverse=True):
        print(f"AUC: {res['auc']:.4f} | Params: {res['params']}")
    print("="*70)

if __name__ == "__main__":
    run_grid_search()