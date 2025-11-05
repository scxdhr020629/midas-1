import sys
import torch
import torch.nn as nn
from numpy import interp
import numpy as np
from model_1 import AblationGCNNetmuti  # Import your ablation model file

from utils import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from torch_geometric.data import DataLoader
import os
import random
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Define LOG_INTERVAL globally
LOG_INTERVAL = 45

# Define loss_fn globally
loss_fn = nn.BCELoss()


def train(model, device, train_loader, optimizer, epoch):
    """训练一个epoch，返回平均损失"""
    model.train()
    epoch_losses = []

    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        output = model(data)

        # 准备好标签
        labels = data.y.view(-1, 1).float().to(device)

        # --- 修复 1: 检查模型输出是否有 NaN ---
        if torch.isnan(output).any():
            print(f"\n[!!! 致命错误：模型输出 NaN !!!]")
            print(f"  错误发生在: Epoch {epoch}, Batch {batch_idx}")
            raise ValueError("Model output is NaN. Stopping training.")

        # --- 修复 2: [关键] 裁剪 output 来防止 log(0) 导致数值不稳定 ---
        epsilon = 1e-7
        output_clamped = torch.clamp(output, min=epsilon, max=1.0 - epsilon)

        # --- 修复 3: 检查标签是否越界 ---
        min_val = labels.min()
        max_val = labels.max()

        if min_val < 0.0 or max_val > 1.0:
            print(f"\n[!!! 致命错误：标签越界 !!!]")
            print(f"  错误发生在: Epoch {epoch}, Batch {batch_idx}")
            print(f"  标签最小值 (min): {min_val.item()}")
            print(f"  标签最大值 (max): {max_val.item()}")
            raise ValueError("Labels out of bounds for BCELoss. Check your data.")

        # 计算损失
        loss = loss_fn(output_clamped, labels)
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())

        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * train_loader.batch_size,
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item()))

    avg_loss = np.mean(epoch_losses)
    return avg_loss


def validate(model, device, val_loader):
    """验证函数 - 计算验证集上的平均损失"""
    model.eval()
    val_losses = []

    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            output = model(data)
            labels = data.y.view(-1, 1).float().to(device)

            epsilon = 1e-7
            output_clamped = torch.clamp(output, min=epsilon, max=1.0 - epsilon)
            loss = loss_fn(output_clamped, labels)
            val_losses.append(loss.item())

    avg_val_loss = np.mean(val_losses)
    return avg_val_loss


def predicting(model, device, loader):
    """预测函数"""
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

    accuracy = accuracy_score(total_labels, total_preds)
    precision = precision_score(total_labels, total_preds)
    recall = recall_score(total_labels, total_preds)
    f1 = f1_score(total_labels, total_preds)

    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    # Evaluation metrics
    roc_auc = roc_auc_score(total_labels, total_probs)
    precision_vals, recall_vals, _ = precision_recall_curve(total_labels, total_probs)
    sorted_indices = np.argsort(recall_vals)
    recall_vals = recall_vals[sorted_indices]
    precision_vals = precision_vals[sorted_indices]

    pr_auc = auc(recall_vals, precision_vals)
    print(f"ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")

    return accuracy, precision, recall, f1, roc_auc, pr_auc


def plot_training_curves(train_losses, val_losses, fold, ablation_mode, save_dir='results'):
    """绘制训练和验证损失曲线"""
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=4)
    plt.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=4)

    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontsize=14, fontweight='bold')
    plt.title(f'Training and Validation Loss - Fold {fold + 1} ({ablation_mode})',
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    # 保存图片
    save_path = os.path.join(save_dir, f'loss_curve_fold{fold + 1}_{ablation_mode}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Loss curve saved to: {save_path}")
    plt.close()

    # 检测过拟合
    min_val_loss = min(val_losses)
    min_val_epoch = val_losses.index(min_val_loss) + 1
    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]

    print(f"\n  📊 Training Analysis:")
    print(f"     - Best Validation Loss: {min_val_loss:.4f} at Epoch {min_val_epoch}")
    print(f"     - Final Training Loss: {final_train_loss:.4f}")
    print(f"     - Final Validation Loss: {final_val_loss:.4f}")
    print(f"     - Train-Val Gap: {abs(final_val_loss - final_train_loss):.4f}")

    # 过拟合判断
    if final_val_loss > min_val_loss * 1.1:
        print(f"     ⚠️  Warning: Possible overfitting detected!")
        print(f"        Validation loss increased from {min_val_loss:.4f} to {final_val_loss:.4f}")
    elif final_train_loss < final_val_loss * 0.8:
        print(f"     ⚠️  Warning: Large train-val gap, possible overfitting!")
    else:
        print(f"     ✅ No obvious overfitting detected.")


def plot_all_folds_curves(all_train_losses, all_val_losses, ablation_mode, save_dir='results'):
    """绘制所有折的训练曲线在一张图上"""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Training and Validation Loss - All Folds ({ablation_mode})',
                 fontsize=18, fontweight='bold')

    # 前5个subplot画每一折
    for fold in range(5):
        row = fold // 3
        col = fold % 3
        ax = axes[row, col]

        epochs = range(1, len(all_train_losses[fold]) + 1)
        ax.plot(epochs, all_train_losses[fold], 'b-o', label='Training', linewidth=2, markersize=3)
        ax.plot(epochs, all_val_losses[fold], 'r-s', label='Validation', linewidth=2, markersize=3)

        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
        ax.set_title(f'Fold {fold + 1}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')

    # 第6个subplot画平均曲线
    ax = axes[1, 2]
    epochs = range(1, len(all_train_losses[0]) + 1)

    # 计算平均和标准差
    avg_train = np.mean(all_train_losses, axis=0)
    avg_val = np.mean(all_val_losses, axis=0)
    std_train = np.std(all_train_losses, axis=0)
    std_val = np.std(all_val_losses, axis=0)

    ax.plot(epochs, avg_train, 'b-o', label='Avg Training', linewidth=2, markersize=3)
    ax.plot(epochs, avg_val, 'r-s', label='Avg Validation', linewidth=2, markersize=3)
    ax.fill_between(epochs, avg_train - std_train, avg_train + std_train, alpha=0.2, color='blue')
    ax.fill_between(epochs, avg_val - std_val, avg_val + std_val, alpha=0.2, color='red')

    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax.set_title('Average Across All Folds', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'all_folds_loss_curves_{ablation_mode}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ All folds loss curves saved to: {save_path}")
    plt.close()


def save_training_history(all_train_losses, all_val_losses, all_metrics, ablation_mode, save_dir='results'):
    """保存训练历史到JSON文件"""
    os.makedirs(save_dir, exist_ok=True)

    history = {
        'ablation_mode': ablation_mode,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'folds': []
    }

    for fold in range(5):
        fold_history = {
            'fold': fold + 1,
            'train_losses': [float(x) for x in all_train_losses[fold]],
            'val_losses': [float(x) for x in all_val_losses[fold]],
            'final_metrics': {
                'accuracy': float(all_metrics['accuracies'][fold]),
                'precision': float(all_metrics['precisions'][fold]),
                'recall': float(all_metrics['recalls'][fold]),
                'f1': float(all_metrics['f1_scores'][fold]),
                'roc_auc': float(all_metrics['roc_aucs'][fold]),
                'pr_auc': float(all_metrics['pr_aucs'][fold])
            }
        }
        history['folds'].append(fold_history)

    # 添加平均指标
    history['average_metrics'] = {
        'accuracy': f"{np.mean(all_metrics['accuracies']):.4f} ± {np.std(all_metrics['accuracies']):.4f}",
        'precision': f"{np.mean(all_metrics['precisions']):.4f} ± {np.std(all_metrics['precisions']):.4f}",
        'recall': f"{np.mean(all_metrics['recalls']):.4f} ± {np.std(all_metrics['recalls']):.4f}",
        'f1': f"{np.mean(all_metrics['f1_scores']):.4f} ± {np.std(all_metrics['f1_scores']):.4f}",
        'roc_auc': f"{np.mean(all_metrics['roc_aucs']):.4f} ± {np.std(all_metrics['roc_aucs']):.4f}",
        'pr_auc': f"{np.mean(all_metrics['pr_aucs']):.4f} ± {np.std(all_metrics['pr_aucs']):.4f}"
    }

    save_path = os.path.join(save_dir, f'training_history_{ablation_mode}.json')
    with open(save_path, 'w') as f:
        json.dump(history, f, indent=4)

    print(f"✅ Training history saved to: {save_path}")


# ==================== Main Training Script ====================

modeling = AblationGCNNetmuti

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
# NUM_EPOCHS = 50
NUM_EPOCHS = 30
NUM_FOLDS = 5

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

print('\nRunning on ', model_st + '_' + ablation_mode)

# Create results directory
os.makedirs('results', exist_ok=True)

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

# Storage for training curves
all_train_losses = []
all_val_losses = []

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

    # Storage for this fold's losses
    train_losses = []
    val_losses = []

    print(f"\nTraining on {len(train_loader.dataset)} samples...")
    print(f"Validating on {len(test_loader.dataset)} samples...")

    # Training epochs
    for epoch in range(NUM_EPOCHS):
        # Train
        train_loss = train(model, device, train_loader, optimizer, epoch + 1)
        train_losses.append(train_loss)

        # Validate
        val_loss = validate(model, device, test_loader)
        val_losses.append(val_loss)

        # Print epoch summary
        print(f'  📈 Epoch {epoch + 1:2d}/{NUM_EPOCHS}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}')

    # Store losses for all folds
    all_train_losses.append(train_losses)
    all_val_losses.append(val_losses)

    # Plot training curve for this fold
    print(f"\n  📊 Plotting training curves for Fold {fold + 1}...")
    plot_training_curves(train_losses, val_losses, fold, ablation_mode)

    # Final prediction on test set
    print(f"\n  🎯 Final evaluation on test set:")
    accuracy, precision, recall, f1, roc_auc, pr_auc = predicting(model, device, test_loader)

    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    roc_aucs.append(roc_auc)
    pr_aucs.append(pr_auc)

# ==================== Plot All Folds Summary ====================

print(f"\n{'=' * 70}")
print("Generating summary plots...")
print(f"{'=' * 70}")

plot_all_folds_curves(all_train_losses, all_val_losses, ablation_mode)

# ==================== Save Training History ====================

all_metrics = {
    'accuracies': accuracies,
    'precisions': precisions,
    'recalls': recalls,
    'f1_scores': f1_scores,
    'roc_aucs': roc_aucs,
    'pr_aucs': pr_aucs
}

save_training_history(all_train_losses, all_val_losses, all_metrics, ablation_mode)

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
print(f"   - Individual fold loss curves: loss_curve_fold*.png")
print(f"   - Summary of all folds: all_folds_loss_curves_{ablation_mode}.png")
print(f"   - Training history JSON: training_history_{ablation_mode}.json")
print("\n🎉 Training completed successfully!")