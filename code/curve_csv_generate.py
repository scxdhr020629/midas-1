import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd  # ✨ 新增：用于保存CSV
from model_fixed import AttnFusionGCNNet
from utils import *
# ✨ 新增 roc_curve
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, precision_recall_curve, auc, matthews_corrcoef, confusion_matrix, roc_curve
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
print(f"[Config] Metrics: Acc, MCC, Sen, Spe, AUC, AUPR")
print(f"{'='*70}\n")

# ============================================================================
# 训练函数
# ============================================================================
def get_contrastive_weight(epoch, warmup_epochs=5):
    """
    保留 Loss 权重的 Warmup 逻辑
    """
    if epoch <= warmup_epochs:
        progress = epoch / warmup_epochs
        return 0.5 * (1 - np.cos(np.pi * progress))
    return 1.0

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    metrics = {
        'total_loss': 0,
        'bce_loss': 0,
        'mirna_cl_loss': 0,
        'drug_cl_loss': 0
    }
    batch_count = 0
    contrastive_weight_factor = get_contrastive_weight(epoch, WARMUP_EPOCHS)

    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)

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
        
        loss_bce = loss_fn(output, labels)
        loss_mirna_contrastive = loss_dict['contrastive_mirna']
        loss_drug_contrastive = loss_dict['contrastive_drug']

        loss = (GAMMA * loss_bce +
                contrastive_weight_factor * (ALPHA * loss_mirna_contrastive +
                                             BETA * loss_drug_contrastive))

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n[Error] Loss is NaN/Inf at Epoch {epoch}, Batch {batch_idx}")
            return None

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        metrics['total_loss'] += loss.item()
        metrics['bce_loss'] += loss_bce.item()
        metrics['mirna_cl_loss'] += loss_mirna_contrastive.item()
        metrics['drug_cl_loss'] += loss_drug_contrastive.item()
        batch_count += 1

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

    # --- ✨ 计算新的指标集 ✨ ---
    
    # 1. Accuracy
    acc = accuracy_score(total_labels, total_preds)
    
    # 2. MCC (Matthews Correlation Coefficient)
    mcc = matthews_corrcoef(total_labels, total_preds)
    
    # 3. Sensitivity (Recall) & Specificity
    # 混淆矩阵: tn, fp, fn, tp
    # 为了防止某些 batch 只有一类导致 unpack 错误，指定 labels=[0, 1]
    cm = confusion_matrix(total_labels, total_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    sen = recall_score(total_labels, total_preds, zero_division=0) # Recall
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # Specificity

    # 4. AUC
    try:
        roc_auc = roc_auc_score(total_labels, total_probs)
    except ValueError:
        roc_auc = 0.5

    # 5. AUPR
    precision_vals, recall_vals, _ = precision_recall_curve(total_labels, total_probs)
    pr_auc = auc(recall_vals, precision_vals)

    # ✨ 修改返回值：额外返回 labels 和 probs 用于后续画图
    return acc, mcc, sen, spe, roc_auc, pr_auc, total_labels, total_probs

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
        'acc': [], 'mcc': [], 'sen': [], 'spe': [], 'auc': [], 'pr_auc': []
    }
    
    # ✨ 新增：存储每一折的原始预测结果
    all_fold_labels = []
    all_fold_probs = []

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

        # ✨ 更新打印表头
        print(f"{'Epoch':<5} | {'TotLoss':<7} {'BCE':<7} {'miR_CL':<7} {'Drug_CL':<7} | {'AUC':<7} {'AUPR':<7} {'Acc':<7} {'MCC':<7} {'Sen':<7} {'Spe':<7}")
        print("-" * 115)

        for epoch in range(1, NUM_EPOCHS + 1):
            train_metrics = train(model, device, train_loader, optimizer, epoch)
            
            if train_metrics is None: 
                break

            scheduler.step()
            
            # 获取新指标 (注意这里加了两个 _ 来接收不需要的 labels 和 probs)
            acc, mcc, sen, spe, auc_score, pr_auc_score, _, _ = predicting(model, device, test_loader)
            
            # ✨ 更新行打印格式
            print(f"{epoch:<5} | "
                  f"{train_metrics['total_loss']:.4f}  {train_metrics['bce_loss']:.4f}  "
                  f"{train_metrics['mirna_cl_loss']:.4f}  {train_metrics['drug_cl_loss']:.4f}   | "
                  f"{auc_score:.4f}  {pr_auc_score:.4f}  {acc:.4f}  {mcc:.4f}  {sen:.4f}  {spe:.4f}")

        # --- Fold 结束 ---
        print(f"\n[Fold {fold + 1} Final] (Using Last Epoch Model)")
        # ✨ 这里我们需要接收 labels 和 probs
        acc, mcc, sen, spe, auc_score, pr_auc_score, fold_labels, fold_probs = predicting(model, device, test_loader)
        
        # 收集数据
        all_fold_labels.append(fold_labels)
        all_fold_probs.append(fold_probs)
        
        print(f"Result -> Acc: {acc:.4f}, MCC: {mcc:.4f}, Sen: {sen:.4f}, Spe: {spe:.4f}, AUC: {auc_score:.4f}, AUPR: {pr_auc_score:.4f}")

        metrics_history['acc'].append(acc)
        metrics_history['mcc'].append(mcc)
        metrics_history['sen'].append(sen)
        metrics_history['spe'].append(spe)
        metrics_history['auc'].append(auc_score)
        metrics_history['pr_auc'].append(pr_auc_score)

    print("\n" + "="*80)
    print("FINAL 5-FOLD CV RESULTS (Mean ± Std)")
    print("="*80)
    print(f"Acc:  {np.mean(metrics_history['acc']):.4f} ± {np.std(metrics_history['acc']):.4f}")
    print(f"MCC:  {np.mean(metrics_history['mcc']):.4f} ± {np.std(metrics_history['mcc']):.4f}")
    print(f"Sen:  {np.mean(metrics_history['sen']):.4f} ± {np.std(metrics_history['sen']):.4f}")
    print(f"Spe:  {np.mean(metrics_history['spe']):.4f} ± {np.std(metrics_history['spe']):.4f}")
    print(f"AUC:  {np.mean(metrics_history['auc']):.4f} ± {np.std(metrics_history['auc']):.4f}")
    print(f"AUPR: {np.mean(metrics_history['pr_auc']):.4f} ± {np.std(metrics_history['pr_auc']):.4f}")
    print("="*80)

    # ============================================================================
    # ✨ 结果生成与保存：Mean ROC & Mean PR Curves
    # ============================================================================
    
    # 1. 生成 Mean ROC Curve 数据
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    
    for y_true, y_score in zip(all_fold_labels, all_fold_probs):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0 # 确保终点是1
    
    # 保存 ROC CSV
    df_roc = pd.DataFrame({'mean_fpr': mean_fpr, 'mean_tpr': mean_tpr})
    df_roc.to_csv('MDCL_mean_fpr_tpr_resistance.csv', index=False)
    print("\n[Output] ROC curve data saved to 'MDCL_mean_fpr_tpr_resistance.csv'")
    
    # 2. 生成 Mean PR Curve 数据
    # PR 曲线插值通常在 recall 轴上进行 (0到1)
    mean_recall = np.linspace(0, 1, 100)
    precisions = []
    
    for y_true, y_score in zip(all_fold_labels, all_fold_probs):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        # precision_recall_curve 返回的 recall 通常是递减的，interp 需要 x 递增
        # 所以我们需要反转
        reversed_recall = recall[::-1]
        reversed_precision = precision[::-1]
        
        interp_precision = np.interp(mean_recall, reversed_recall, reversed_precision)
        precisions.append(interp_precision)
        
    mean_precision = np.mean(precisions, axis=0)
    
    # 保存 PR CSV
    df_pr = pd.DataFrame({'mean_recall': mean_recall, 'mean_precision': mean_precision})
    df_pr.to_csv('MDCL_mean_recall_precision_resistance.csv', index=False)
    print("[Output] PR curve data saved to 'MDCL_mean_recall_precision_resistance.csv'")

    # 3. 计算并打印 Mean Curve 下的 AUC (作为最终参考)
    # 注意：这里计算的是"平均曲线的AUC"，可能与"各折AUC的平均值"微小不同，但通常用于画图很准确
    final_roc_auc = auc(mean_fpr, mean_tpr)
    final_pr_auc = auc(mean_recall, mean_precision)
    
    print(f"\n[Final Control Output]")
    print(f"AUC (Mean Curve): {final_roc_auc:.4f}")
    print(f"AUPR (Mean Curve): {final_pr_auc:.4f}")