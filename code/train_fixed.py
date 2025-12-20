import sys
import torch
import torch.nn as nn
import numpy as np
from model_fixed import AttnFusionGCNNet
from utils import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from torch_geometric.loader import DataLoader
import os
import time

# --- 全局常量 ---
LOG_INTERVAL = 45
NUM_FOLDS = 5
loss_fn = nn.BCELoss()

# ============================================================================
# ✨ 超参数配置 (Standard CCL - 无动态采样)
# ============================================================================
LR = 0.0005
WEIGHT_DECAY = 0.0032
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
NUM_EPOCHS = 30
WARMUP_EPOCHS = 5

# --- 对比学习参数 ---
ALPHA = 0.5
BETA = 0.5
GAMMA = 1.0
TEMPERATURE = 0.1
LAM = 0.5
CONTRASTIVE_DIM = 128

print(f"{'='*70}")
print(f"[Config] Training Mode: Standard CCL (ASPS Removed)")
print(f"[Config] Total Epochs: {NUM_EPOCHS}")
print(f"[Config] Warmup Epochs (for Loss Weight): {WARMUP_EPOCHS}")
print(f"[Config] Model Selection: 最后一轮 (Epoch {NUM_EPOCHS})")
print(f"{'='*70}\n")

# ============================================================================
# 训练函数
# ============================================================================
def get_contrastive_weight(epoch, warmup_epochs=5):
    """
    保留 Loss 权重的 Warmup 逻辑，防止早期对比 Loss 干扰主任务
    """
    if epoch <= warmup_epochs:
        progress = epoch / warmup_epochs
        return 0.5 * (1 - np.cos(np.pi * progress))
    return 1.0

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    # 初始化累加器
    metrics = {
        'total_loss': 0,
        'bce_loss': 0,
        'mirna_cl_loss': 0,
        'drug_cl_loss': 0
    }
    batch_count = 0
    # 获取当前 epoch 的对比损失权重
    contrastive_weight_factor = get_contrastive_weight(epoch, WARMUP_EPOCHS)

    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)

        # 参数传入保持兼容性，但内部已不再使用 epoch 进行动态采样
        output, loss_dict = model(
            data,
            current_epoch=epoch,
            total_epochs=NUM_EPOCHS,
            warmup_epochs=WARMUP_EPOCHS,
            return_contrastive_loss=True
        )

        labels = data.y.view(-1, 1).float().to(device)
        output = output.view(-1, 1)
        output = torch.clamp(output, min=1e-7, max=1.0 - 1e-7)
        
        # 计算各项 Loss
        loss_bce = loss_fn(output, labels)
        loss_mirna_contrastive = loss_dict['contrastive_mirna']
        loss_drug_contrastive = loss_dict['contrastive_drug']

        # 总 Loss
        loss = (GAMMA * loss_bce +
                contrastive_weight_factor * (ALPHA * loss_mirna_contrastive +
                                             BETA * loss_drug_contrastive))

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n[Error] Loss is NaN/Inf at Epoch {epoch}, Batch {batch_idx}")
            return None

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        # 记录数值
        metrics['total_loss'] += loss.item()
        metrics['bce_loss'] += loss_bce.item()
        metrics['mirna_cl_loss'] += loss_mirna_contrastive.item()
        metrics['drug_cl_loss'] += loss_drug_contrastive.item()
        batch_count += 1

    # 计算平均值
    return {k: v / batch_count for k, v in metrics.items()}

def predicting(model, device, loader):
    model.eval()
    total_probs = []
    total_labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(
                data,
                current_epoch=0,
                total_epochs=NUM_EPOCHS,
                warmup_epochs=WARMUP_EPOCHS,
                return_contrastive_loss=False
            )
            probs = output.cpu().numpy().flatten()
            total_probs.extend(probs)
            total_labels.extend(data.y.view(-1, 1).cpu().numpy().flatten())

    total_probs = np.array(total_probs)
    total_labels = np.array(total_labels)
    total_preds = (total_probs >= 0.5).astype(int)

    # 计算各项指标
    accuracy = accuracy_score(total_labels, total_preds)
    precision = precision_score(total_labels, total_preds, zero_division=0)
    recall = recall_score(total_labels, total_preds, zero_division=0)
    f1 = f1_score(total_labels, total_preds, zero_division=0)

    try:
        roc_auc = roc_auc_score(total_labels, total_probs)
    except ValueError:
        roc_auc = 0.5

    precision_vals, recall_vals, _ = precision_recall_curve(total_labels, total_probs)
    pr_auc = auc(recall_vals, precision_vals)

    return accuracy, precision, recall, f1, roc_auc, pr_auc

# ============================================================================
# 主程序
# ============================================================================
if __name__ == "__main__":
    cuda_name = "cuda:0"
    if len(sys.argv) > 1:
        cuda_name = "cuda:" + str(int(sys.argv[1]))

    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}\n')
    
    metrics_history = {
        'acc': [], 'prec': [], 'rec': [], 'f1': [], 'auc': [], 'pr_auc': []
    }

    for fold in range(NUM_FOLDS):
        print(f"\n{'='*70}")
        print(f">>> Fold {fold + 1}/{NUM_FOLDS}")
        print(f"{'='*70}")

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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=LR * 0.01)

        print(f"{'Epoch':<5} | {'TotLoss':<7} {'BCE':<7} {'miR_CL':<7} {'Drug_CL':<7} | {'AUC':<7} {'AUPR':<7} {'F1':<7} {'Acc':<7} {'Rec':<7}")
        print("-" * 100)

        for epoch in range(1, NUM_EPOCHS + 1):
            # 1. 训练并获取详细 Loss
            train_metrics = train(model, device, train_loader, optimizer, epoch)
            
            if train_metrics is None: # Loss NaN check
                break

            scheduler.step()
            
            # 2. 验证并获取详细指标
            acc, prec, rec, f1, auc_score, pr_auc_score = predicting(model, device, test_loader)
            
            # 3. 详细打印
            print(f"{epoch:<5} | "
                  f"{train_metrics['total_loss']:.4f}  {train_metrics['bce_loss']:.4f}  "
                  f"{train_metrics['mirna_cl_loss']:.4f}  {train_metrics['drug_cl_loss']:.4f}   | "
                  f"{auc_score:.4f}  {pr_auc_score:.4f}  {f1:.4f}  {acc:.4f}  {rec:.4f}")

        # --- Fold 结束 ---
        print(f"\n[Fold {fold + 1} Final] (Using Last Epoch Model)")
        acc, prec, rec, f1, auc_score, pr_auc_score = predicting(model, device, test_loader)
        
        print(f"Result -> AUC: {auc_score:.4f}, AUPR: {pr_auc_score:.4f}, F1: {f1:.4f}, Acc: {acc:.4f}, Rec: {rec:.4f}")

        metrics_history['acc'].append(acc)
        metrics_history['prec'].append(prec)
        metrics_history['rec'].append(rec)
        metrics_history['f1'].append(f1)
        metrics_history['auc'].append(auc_score)
        metrics_history['pr_auc'].append(pr_auc_score)

    print("\n" + "="*80)
    print("FINAL 5-FOLD CV RESULTS (Mean ± Std)")
    print("="*80)
    print(f"AUC:       {np.mean(metrics_history['auc']):.4f} ± {np.std(metrics_history['auc']):.4f}")
    print(f"AUPR:      {np.mean(metrics_history['pr_auc']):.4f} ± {np.std(metrics_history['pr_auc']):.4f}")
    print(f"Accuracy:  {np.mean(metrics_history['acc']):.4f} ± {np.std(metrics_history['acc']):.4f}")
    print(f"F1-Score:  {np.mean(metrics_history['f1']):.4f} ± {np.std(metrics_history['f1']):.4f}")
    print(f"Recall:    {np.mean(metrics_history['rec']):.4f} ± {np.std(metrics_history['rec']):.4f}")
    print(f"Precision: {np.mean(metrics_history['prec']):.4f} ± {np.std(metrics_history['prec']):.4f}")
    print("="*80)