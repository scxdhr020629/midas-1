import sys
import torch
import torch.nn as nn
import numpy as np
from model_1 import AttnFusionGCNNet  # <-- 假设 AttnFusionGCNNet 在 model_1.py 中
from utils import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
from torch_geometric.data import DataLoader
import os
import time

# --- 全局常量 ---
LOG_INTERVAL = 45
NUM_FOLDS = 5
loss_fn = nn.BCELoss()

# ============================================================================
#
# ✨ Optuna 找到的最佳超参数 ✨
#
# ============================================================================
LR = 0.0005
WEIGHT_DECAY = 0.0032
TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128  # 保持一致
NUM_EPOCHS = 45


# ============================================================================


# ============================================================================
#
# 核心训练/预测函数
#
# ============================================================================

def train(model, device, train_loader, optimizer, epoch):
    """训练一个epoch"""
    print(f'Training epoch: {epoch}...')  # <-- 恢复了日志
    model.train()
    total_loss = 0
    batch_count = 0
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        output = model(data)
        labels = data.y.view(-1, 1).float().to(device)

        if torch.isnan(output).any():
            print(f"\n[!!! 致命错误：模型输出 NaN !!!]")
            print(f"  错误发生在: Epoch {epoch}, Batch {batch_idx}")
            raise ValueError("Model output is NaN. Stopping training.")

        epsilon = 1e-7
        output_clamped = torch.clamp(output, min=epsilon, max=1.0 - epsilon)

        if labels.min() < 0.0 or labels.max() > 1.0:
            print(f"\n[!!! 致命错误：标签越界 !!!]")
            print(f"  错误发生在: Epoch {epoch}, Batch {batch_idx}")
            raise ValueError("Labels out of bounds for BCELoss. Check your data.")

        loss = loss_fn(output_clamped, labels)

        if torch.isnan(loss):
            print(f"\n[!!! 致命错误：损失(Loss)为 NaN !!!]")
            print(f"  错误发生在: Epoch {epoch}, Batch {batch_idx}")
            raise ValueError("Loss is NaN. Stopping training.")

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batch_count += 1

        # 恢复了 Batch 日志
        if batch_idx % LOG_INTERVAL == 0:
            print('  Batch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx,
                batch_idx * train_loader.batch_size,
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item()))

    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    print(f'Epoch {epoch} - Average Loss: {avg_loss:.6f}')


def predicting(model, device, loader):
    """预测函数 - 只返回指标"""
    model.eval()
    total_probs = []
    total_labels = []

    print('Making prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            probs = output.cpu().numpy()
            total_probs.extend(probs)
            total_labels.extend(data.y.view(-1, 1).cpu().numpy())

    total_probs = np.array(total_probs).flatten()
    total_labels = np.array(total_labels).flatten()
    total_preds = (total_probs >= 0.5).astype(int)

    accuracy = accuracy_score(total_labels, total_preds)
    precision = precision_score(total_labels, total_preds, zero_division=0)
    recall = recall_score(total_labels, total_preds, zero_division=0)
    f1 = f1_score(total_labels, total_preds, zero_division=0)

    try:
        roc_auc = roc_auc_score(total_labels, total_probs)
    except ValueError:
        print("Warning: Only one class present in y_true. ROC AUC score set to 0.5.")
        roc_auc = 0.5

    precision_vals, recall_vals, _ = precision_recall_curve(total_labels, total_probs)
    pr_auc = auc(recall_vals, precision_vals)

    return accuracy, precision, recall, f1, roc_auc, pr_auc


# ============================================================================
#
# Main Execution (5-Fold CV)
# (移除了所有 Optuna, run_cv_training, 和 objective 函数)
#
# ============================================================================

if __name__ == "__main__":

    # --- 1. 基本设置 ---
    cuda_name = "cuda:0"
    if len(sys.argv) > 1:
        cuda_name = "cuda:" + str(int(sys.argv[1]))
    print('cuda_name:', cuda_name)

    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}\n')

    print(f"Running 5-Fold CV with OPTIMIZED Hyperparameters...")
    modeling = AttnFusionGCNNet
    print(f"Model: {modeling.__name__}")
    print(f"Learning Rate (LR): {LR}")
    print(f"Weight Decay: {WEIGHT_DECAY}")
    print(f"Epochs per fold: {NUM_EPOCHS}")
    print(f"Batch Size: {TRAIN_BATCH_SIZE}")

    # --- 2. 存储结果 ---
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    roc_aucs = []
    pr_aucs = []

    start_time_total = time.time()

    # --- 3. 5-Fold CV 循环 ---
    for fold in range(NUM_FOLDS):
        print(f"\n{'=' * 70}")
        print(f"Fold {fold + 1}/{NUM_FOLDS}")
        print(f"{'=' * 70}")

        fold_start_time = time.time()

        train_data = TestbedDataset(root='data', dataset='train' + str(fold))
        test_data = TestbedDataset(root='data', dataset='test' + str(fold))
        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, drop_last=True)

        model = modeling().to(device)

        # 使用 Optuna 找到的最佳参数
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        for epoch in range(1, NUM_EPOCHS + 1):
            train(model, device, train_loader, optimizer, epoch)

        accuracy, precision, recall, f1, roc_auc, pr_auc = predicting(model, device, test_loader)

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        roc_aucs.append(roc_auc)
        pr_aucs.append(pr_auc)

        fold_end_time = time.time()
        print(f"\nFold {fold + 1} Results:")
        print(f"  Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")
        print(f"  Fold completed in {(fold_end_time - fold_start_time) / 60:.2f} minutes.")

    # --- 4. 打印最终总结 ---

    end_time_total = time.time()
    print("\n" + "=" * 70)
    print(f"FINAL RESULTS - {NUM_FOLDS}-Fold Cross-Validation (Optimized)")
    print(f"Total time: {(end_time_total - start_time_total) / 60:.2f} minutes")
    print("=" * 70)
    print(f"Accuracy:  {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
    print(f"Recall:    {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    print(f"F1 Score:  {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print(f"ROC AUC:   {np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f}")
    print(f"PR AUC:    {np.mean(pr_aucs):.4f} ± {np.std(pr_aucs):.4f}")
    print("=" * 70)

    print("\nIndividual Fold Results:")
    print("-" * 70)
    for i in range(NUM_FOLDS):
        print(f"Fold {i + 1}: Acc={accuracies[i]:.4f}, Prec={precisions[i]:.4f}, "
              f"Rec={recalls[i]:.4f}, F1={f1_scores[i]:.4f}, "
              f"ROC-AUC={roc_aucs[i]:.4f}, PR-AUC={pr_aucs[i]:.4f}")
    print("-" * 70)
    print("\n🎉 Training script with optimized parameters finished.")