import sys
import torch
import torch.nn as nn
from numpy import interp
import numpy as np
from model_1 import AblationGCNNetmuti  # Import your ablation model file

from utils import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
from torch_geometric.data import DataLoader
import os
import random
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端，适合服务器环境

# 创建结果保存目录
os.makedirs('results/roc_curves', exist_ok=True)
os.makedirs('results/pr_curves', exist_ok=True)
os.makedirs('results/combined_curves', exist_ok=True)
os.makedirs('results/metrics', exist_ok=True)

# Define LOG_INTERVAL globally
LOG_INTERVAL = 45

# Define loss_fn globally
loss_fn = nn.BCELoss()


def train(model, device, train_loader, optimizer, epoch):
    """训练一个epoch"""
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        output = model(data)

        # 准备好标签
        labels = data.y.view(-1, 1).float().to(device)

        # --- 修复 1: 检查模型输出是否有 NaN ---
        if torch.isnan(output).any():
            print(f"\n[!!! 致命错误：模型输出 NaN !!!]")
            print(f"  错误发生在: Batch {batch_idx}")
            raise ValueError("Model output is NaN. Stopping training.")

        # --- 修复 2: [关键] 裁剪 output 来防止 log(0) 导致数值不稳定 ---
        epsilon = 1e-7
        output_clamped = torch.clamp(output, min=epsilon, max=1.0 - epsilon)

        # --- 修复 3: 检查标签是否越界 ---
        min_val = labels.min()
        max_val = labels.max()

        if min_val < 0.0 or max_val > 1.0:
            print(f"\n[!!! 致命错误：标签越界 !!!]")
            print(f"  错误发生在: Batch {batch_idx}")
            print(f"  标签最小值 (min): {min_val.item()}")
            print(f"  标签最大值 (max): {max_val.item()}")
            raise ValueError("Labels out of bounds for BCELoss. Check your data.")

        # 计算损失
        loss = loss_fn(output_clamped, labels)
        loss.backward()
        optimizer.step()

        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * train_loader.batch_size,
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item()))


def predicting(model, device, loader):
    """预测函数 - 返回概率、预测和标签"""
    model.eval()
    total_probs = []
    total_preds = []
    total_labels = []

    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            probs = output.cpu().numpy()
            preds = (output >= 0.5).float().cpu().numpy()

            total_probs.extend(probs)
            total_preds.extend(preds)
            total_labels.extend(data.y.view(-1, 1).cpu().numpy())

    total_probs = np.array(total_probs).flatten()
    total_preds = np.array(total_preds).flatten()
    total_labels = np.array(total_labels).flatten()

    # 计算指标
    accuracy = accuracy_score(total_labels, total_preds)
    precision = precision_score(total_labels, total_preds)
    recall = recall_score(total_labels, total_preds)
    f1 = f1_score(total_labels, total_preds)

    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    # ROC-AUC
    roc_auc = roc_auc_score(total_labels, total_probs)

    # PR-AUC
    precision_vals, recall_vals, _ = precision_recall_curve(total_labels, total_probs)
    sorted_indices = np.argsort(recall_vals)
    recall_vals = recall_vals[sorted_indices]
    precision_vals = precision_vals[sorted_indices]
    pr_auc = auc(recall_vals, precision_vals)

    print(f"ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")

    return accuracy, precision, recall, f1, roc_auc, pr_auc, total_labels, total_probs


def plot_roc_curve(labels, probs, fold, ablation_mode, save_dir='results/roc_curves'):
    """绘制单个fold的ROC曲线"""
    os.makedirs(save_dir, exist_ok=True)

    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    # 找到最佳阈值点（Youden's Index）
    youden_index = tpr - fpr
    best_threshold_idx = np.argmax(youden_index)
    best_threshold = thresholds[best_threshold_idx]
    best_fpr = fpr[best_threshold_idx]
    best_tpr = tpr[best_threshold_idx]

    # 绘图
    plt.figure(figsize=(10, 8))

    # ROC曲线
    plt.plot(fpr, tpr, color='darkorange', lw=2.5,
             label=f'ROC Curve (AUC = {roc_auc:.4f})')

    # 对角线（随机分类器）
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random Classifier (AUC = 0.5000)')

    # 标记最佳阈值点
    plt.plot(best_fpr, best_tpr, 'ro', markersize=10,
             label=f'Best Threshold = {best_threshold:.3f}\n(TPR={best_tpr:.3f}, FPR={best_fpr:.3f})')

    # 设置坐标轴
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    plt.title(f'ROC Curve - Fold {fold + 1} ({ablation_mode})', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')

    # 添加文本注释
    plt.text(0.6, 0.2, f'AUC = {roc_auc:.4f}', fontsize=14,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'roc_fold{fold + 1}_{ablation_mode}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✅ ROC curve saved to: {save_path}")

    return roc_auc, best_threshold


def plot_pr_curve(labels, probs, fold, ablation_mode, save_dir='results/pr_curves'):
    """绘制单个fold的Precision-Recall曲线"""
    os.makedirs(save_dir, exist_ok=True)

    # 计算PR曲线
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    pr_auc = auc(recall, precision)

    # 计算基线（随机分类器的precision）
    pos_ratio = np.sum(labels) / len(labels)

    # 找到F1最大的点
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_idx]
    best_precision = precision[best_f1_idx]
    best_recall = recall[best_f1_idx]
    best_f1 = f1_scores[best_f1_idx]

    # 绘图
    plt.figure(figsize=(10, 8))

    # PR曲线
    plt.plot(recall, precision, color='blue', lw=2.5,
             label=f'PR Curve (AUC = {pr_auc:.4f})')

    # 基线（随机分类器）
    plt.plot([0, 1], [pos_ratio, pos_ratio], color='red', lw=2, linestyle='--',
             label=f'Random Classifier (Precision = {pos_ratio:.4f})')

    # 标记最佳F1点
    plt.plot(best_recall, best_precision, 'go', markersize=10,
             label=f'Best F1 = {best_f1:.3f}\n(Threshold = {best_threshold:.3f})\n(P={best_precision:.3f}, R={best_recall:.3f})')

    # 设置坐标轴
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=14, fontweight='bold')
    plt.ylabel('Precision', fontsize=14, fontweight='bold')
    plt.title(f'Precision-Recall Curve - Fold {fold + 1} ({ablation_mode})', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')

    # 添加文本注释
    plt.text(0.5, 0.95, f'PR-AUC = {pr_auc:.4f}', fontsize=14,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'pr_fold{fold + 1}_{ablation_mode}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✅ PR curve saved to: {save_path}")

    return pr_auc, best_threshold, best_f1


def plot_combined_curves(labels, probs, fold, ablation_mode, save_dir='results/combined_curves'):
    """绘制ROC和PR曲线在同一张图上（左右布局）"""
    os.makedirs(save_dir, exist_ok=True)

    # 计算ROC
    fpr, tpr, roc_thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    # 计算PR
    precision, recall, pr_thresholds = precision_recall_curve(labels, probs)
    pr_auc = auc(recall, precision)

    # 基线
    pos_ratio = np.sum(labels) / len(labels)

    # 创建2x1的子图
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ========== 左图：ROC曲线 ==========
    ax1 = axes[0]
    ax1.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random (AUC = 0.5000)')

    # 最佳阈值点
    youden_index = tpr - fpr
    best_roc_idx = np.argmax(youden_index)
    ax1.plot(fpr[best_roc_idx], tpr[best_roc_idx], 'ro', markersize=10,
             label=f'Best Point (Threshold={roc_thresholds[best_roc_idx]:.3f})')

    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    ax1.set_title(f'ROC Curve - Fold {fold + 1}', fontsize=14, fontweight='bold')
    ax1.legend(loc="lower right", fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # ========== 右图：PR曲线 ==========
    ax2 = axes[1]
    ax2.plot(recall, precision, color='blue', lw=2.5, label=f'PR Curve (AUC = {pr_auc:.4f})')
    ax2.plot([0, 1], [pos_ratio, pos_ratio], color='red', lw=2, linestyle='--',
             label=f'Random (Precision = {pos_ratio:.4f})')

    # 最佳F1点
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    ax2.plot(recall[best_f1_idx], precision[best_f1_idx], 'go', markersize=10,
             label=f'Best F1={f1_scores[best_f1_idx]:.3f} (Threshold={pr_thresholds[best_f1_idx]:.3f})')

    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Precision', fontsize=13, fontweight='bold')
    ax2.set_title(f'Precision-Recall Curve - Fold {fold + 1}', fontsize=14, fontweight='bold')
    ax2.legend(loc="lower left", fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # 总标题
    fig.suptitle(f'ROC and PR Curves - {ablation_mode.upper()}', fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'combined_fold{fold + 1}_{ablation_mode}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✅ Combined curves saved to: {save_path}")


def plot_all_folds_roc(all_labels, all_probs, ablation_mode, save_dir='results/roc_curves'):
    """绘制所有fold的ROC曲线在一张图上"""
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(12, 10))

    # 用于计算平均ROC
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    # 绘制每一折
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for fold in range(len(all_labels)):
        fpr, tpr, _ = roc_curve(all_labels[fold], all_probs[fold])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # 插值到统一的fpr
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

        plt.plot(fpr, tpr, lw=2, alpha=0.6, color=colors[fold],
                 label=f'Fold {fold + 1} (AUC = {roc_auc:.4f})')

    # 绘制对角线
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', alpha=0.8, label='Random')

    # 计算并绘制平均ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    plt.plot(mean_fpr, mean_tpr, color='navy', lw=3,
             label=f'Mean ROC (AUC = {mean_auc:.4f} ± {std_auc:.4f})')

    # 绘制标准差区域
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='navy', alpha=0.2,
                     label='± 1 std. dev.')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    plt.title(f'ROC Curves - All Folds ({ablation_mode.upper()})', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'roc_all_folds_{ablation_mode}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ All folds ROC curve saved to: {save_path}")
    print(f"   Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")


def plot_all_folds_pr(all_labels, all_probs, ablation_mode, save_dir='results/pr_curves'):
    """绘制所有fold的PR曲线在一张图上"""
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(12, 10))

    # 用于计算平均PR
    mean_recall = np.linspace(0, 1, 100)
    precisions_interp = []
    aucs = []

    # 绘制每一折
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for fold in range(len(all_labels)):
        precision, recall, _ = precision_recall_curve(all_labels[fold], all_probs[fold])
        pr_auc = auc(recall, precision)
        aucs.append(pr_auc)

        # 插值（PR曲线从右到左，需要反转）
        precision_interp = np.interp(mean_recall, recall[::-1], precision[::-1])
        precisions_interp.append(precision_interp)

        plt.plot(recall, precision, lw=2, alpha=0.6, color=colors[fold],
                 label=f'Fold {fold + 1} (AUC = {pr_auc:.4f})')

    # 计算基线
    pos_ratio = np.mean([np.sum(labels) / len(labels) for labels in all_labels])
    plt.plot([0, 1], [pos_ratio, pos_ratio], linestyle='--', lw=2, color='gray', alpha=0.8,
             label=f'Random (Precision = {pos_ratio:.4f})')

    # 计算并绘制平均PR
    mean_precision = np.mean(precisions_interp, axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    plt.plot(mean_recall, mean_precision, color='navy', lw=3,
             label=f'Mean PR (AUC = {mean_auc:.4f} ± {std_auc:.4f})')

    # 绘制标准差区域
    std_precision = np.std(precisions_interp, axis=0)
    precision_upper = np.minimum(mean_precision + std_precision, 1)
    precision_lower = np.maximum(mean_precision - std_precision, 0)
    plt.fill_between(mean_recall, precision_lower, precision_upper, color='navy', alpha=0.2,
                     label='± 1 std. dev.')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=14, fontweight='bold')
    plt.ylabel('Precision', fontsize=14, fontweight='bold')
    plt.title(f'Precision-Recall Curves - All Folds ({ablation_mode.upper()})', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'pr_all_folds_{ablation_mode}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ All folds PR curve saved to: {save_path}")
    print(f"   Mean PR-AUC: {mean_auc:.4f} ± {std_auc:.4f}")


def save_curve_data(all_labels, all_probs, ablation_mode, save_dir='results/metrics'):
    """保存曲线数据到CSV，方便后续分析"""
    import csv
    os.makedirs(save_dir, exist_ok=True)

    # 保存每个fold的曲线数据
    for fold in range(len(all_labels)):
        # ROC数据
        fpr, tpr, roc_thresholds = roc_curve(all_labels[fold], all_probs[fold])
        roc_csv = os.path.join(save_dir, f'roc_data_fold{fold + 1}_{ablation_mode}.csv')
        with open(roc_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Threshold', 'FPR', 'TPR'])
            for i in range(len(fpr)):
                thresh = roc_thresholds[i] if i < len(roc_thresholds) else 0
                writer.writerow([thresh, fpr[i], tpr[i]])

        # PR数据
        precision, recall, pr_thresholds = precision_recall_curve(all_labels[fold], all_probs[fold])
        pr_csv = os.path.join(save_dir, f'pr_data_fold{fold + 1}_{ablation_mode}.csv')
        with open(pr_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Threshold', 'Recall', 'Precision'])
            for i in range(len(recall)):
                thresh = pr_thresholds[i] if i < len(pr_thresholds) else 0
                writer.writerow([thresh, recall[i], precision[i]])

    print(f"\n✅ Curve data saved to: {save_dir}")


# ============================================================================
# Main Training Loop
# ============================================================================

modeling = AblationGCNNetmuti  # Use ablation model
model_st = modeling.__name__

cuda_name = "cuda:0"
if len(sys.argv) > 1:
    cuda_name = "cuda:" + str(int(sys.argv[1]))
print('cuda_name:', cuda_name)

# Ablation mode
ablation_mode = sys.argv[2] if len(sys.argv) > 2 else 'baseline'
print(f"Ablation mode: {ablation_mode}")

TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
LR = 0.0001
NUM_EPOCHS = 50
NUM_FOLDS = 5

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)
print('\nRunning on ', model_st + '_' + ablation_mode)

# Device
device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}\n')

# Storage for results
accuracies = []
precisions = []
recalls = []
f1_scores = []
roc_aucs = []
pr_aucs = []

# ✅ Storage for labels and probabilities (for plotting curves)
all_labels = []
all_probs = []

# ==================== Training Loop ====================

for fold in range(NUM_FOLDS):
    print(f"\n{'=' * 70}")
    print(f"Fold {fold + 1}/{NUM_FOLDS}")
    print(f"{'=' * 70}")

    # Load data
    train_data = TestbedDataset(root='data', dataset='train' + str(fold))
    test_data = TestbedDataset(root='data', dataset='test' + str(fold))
    train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, drop_last=True)

    # Initialize model
    model = modeling(ablation_mode=ablation_mode).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.001)

    # Training
    for epoch in range(NUM_EPOCHS):
        train(model, device, train_loader, optimizer, epoch + 1)

    # ✅ Prediction and get probabilities
    accuracy, precision, recall, f1, roc_auc, pr_auc, labels, probs = predicting(model, device, test_loader)

    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    roc_aucs.append(roc_auc)
    pr_aucs.append(pr_auc)

    # ✅ Store labels and probs for curve plotting
    all_labels.append(labels)
    all_probs.append(probs)

    # ✅ Plot individual fold curves
    print(f"\n📊 Generating curves for Fold {fold + 1}...")
    plot_roc_curve(labels, probs, fold, ablation_mode)
    plot_pr_curve(labels, probs, fold, ablation_mode)
    plot_combined_curves(labels, probs, fold, ablation_mode)

# ==================== Plot All Folds Summary ====================

print(f"\n{'=' * 70}")
print("📊 Generating summary curves for all folds...")
print(f"{'=' * 70}")

plot_all_folds_roc(all_labels, all_probs, ablation_mode)
plot_all_folds_pr(all_labels, all_probs, ablation_mode)

# ✅ Save curve data
save_curve_data(all_labels, all_probs, ablation_mode)

# ==================== Final Results ====================

avg_accuracy = np.mean(accuracies)
avg_precision = np.mean(precisions)
avg_recall = np.mean(recalls)
avg_f1 = np.mean(f1_scores)
avg_roc_auc = np.mean(roc_aucs)
avg_pr_auc = np.mean(pr_aucs)

std_accuracy = np.std(accuracies)
std_precision = np.std(precisions)
std_recall = np.std(recalls)
std_f1 = np.std(f1_scores)
std_roc_auc = np.std(roc_aucs)
std_pr_auc = np.std(pr_aucs)

print("\n" + "=" * 70)
print(f"FINAL RESULTS - {NUM_FOLDS}-Fold Cross-Validation")
print(f"Ablation Mode: '{ablation_mode}'")
print("=" * 70)
print(f"Accuracy:  {avg_accuracy:.4f} ± {std_accuracy:.4f}")
print(f"Precision: {avg_precision:.4f} ± {std_precision:.4f}")
print(f"Recall:    {avg_recall:.4f} ± {std_recall:.4f}")
print(f"F1 Score:  {avg_f1:.4f} ± {std_f1:.4f}")
print(f"ROC AUC:   {avg_roc_auc:.4f} ± {std_roc_auc:.4f}")
print(f"PR AUC:    {avg_pr_auc:.4f} ± {std_pr_auc:.4f}")
print("=" * 70)

print("\nIndividual Fold Results:")
print("-" * 70)
for i in range(NUM_FOLDS):
    print(f"Fold {i + 1}: Acc={accuracies[i]:.4f}, Prec={precisions[i]:.4f}, "
          f"Rec={recalls[i]:.4f}, F1={f1_scores[i]:.4f}, "
          f"ROC-AUC={roc_aucs[i]:.4f}, PR-AUC={pr_aucs[i]:.4f}")
print("-" * 70)

print(f"\n✅ All results saved to 'results/' directory")
print(f"   📁 Individual ROC curves: results/roc_curves/roc_fold*.png")
print(f"   📁 Individual PR curves:  results/pr_curves/pr_fold*.png")
print(f"   📁 Combined curves:       results/combined_curves/combined_fold*.png")
print(f"   📁 Summary ROC curve:     results/roc_curves/roc_all_folds_{ablation_mode}.png")
print(f"   📁 Summary PR curve:      results/pr_curves/pr_all_folds_{ablation_mode}.png")
print(f"   📁 Curve data (CSV):      results/metrics/*_data_fold*.csv")
print("\n🎉 Training and visualization completed successfully!")