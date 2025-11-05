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
import random  # For Python random seed


# Define LOG_INTERVAL globally
LOG_INTERVAL = 45

# Define loss_fn globally
loss_fn = nn.BCELoss()

def train(model, device, train_loader, optimizer, epoch):
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
            print(f"  模型输出了 'NaN' (Not a Number)。这通常是由于数值不稳定。")
            raise ValueError("Model output is NaN. Stopping training.")

        # --- 修复 2: [关键] 裁剪 output 来防止 log(0) 导致数值不稳定 ---
        epsilon = 1e-7  # 定义一个极小值，防止 log(0)
        output_clamped = torch.clamp(output, min=epsilon, max=1.0 - epsilon)

        # --- 修复 3: 检查标签是否越界 ---
        min_val = labels.min()
        max_val = labels.max()

        if min_val < 0.0 or max_val > 1.0:
            print(f"\n[!!! 致命错误：标签越界 !!!]")
            print(f"  错误发生在: Batch {batch_idx}")
            print(f"  标签最小值 (min): {min_val.item()}")
            print(f"  标签最大值 (max): {max_val.item()}")
            print(f"  错误原因: nn.BCELoss (二元交叉熵损失) 要求所有标签值必须在 [0.0, 1.0] 范围内。")
            raise ValueError("Labels out of bounds for BCELoss. Check your data.")

        # 计算损失
        loss = loss_fn(output_clamped, labels)
        loss.backward()
        optimizer.step()

        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * train_loader.batch_size,
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))


def predicting(model, device, loader):
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


modeling = AblationGCNNetmuti  # Use ablation model

model_st = modeling.__name__

cuda_name = "cuda:0"
if len(sys.argv) > 1:
    cuda_name = "cuda:" + str(int(sys.argv[1]))
print('cuda_name:', cuda_name)

# Ablation mode: pass as arg (e.g., python train.py 0 no_matrix), default 'baseline'
ablation_mode = sys.argv[2] if len(sys.argv) > 2 else 'baseline'
print(f"Ablation mode: {ablation_mode}")

TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
LR = 0.0001
NUM_EPOCHS = 50
NUM_FOLDS = 5  # Number of folds (0 to 4)

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

print('\nRunning on ', model_st + '_' + ablation_mode)

# train
device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}\n')

accuracies = []
precisions = []
recalls = []
f1_scores = []
roc_aucs = []
pr_aucs = []

for fold in range(NUM_FOLDS):
    print(f"\n{'='*70}")
    print(f"Fold {fold + 1}/{NUM_FOLDS}")
    print(f"{'='*70}")

    # Load data for this fold (train{fold}/test{fold})
    train_data = TestbedDataset(root='data', dataset='train' + str(fold))
    test_data = TestbedDataset(root='data', dataset='test' + str(fold))
    train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, drop_last=True)

    model = modeling(ablation_mode=ablation_mode).to(device)  # Pass mode to ablation model
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.001)

    for epoch in range(NUM_EPOCHS):
        train(model, device, train_loader, optimizer, epoch + 1)

    accuracy, precision, recall, f1, roc_auc, pr_auc = predicting(model, device, test_loader)
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    roc_aucs.append(roc_auc)
    pr_aucs.append(pr_auc)

# Average over all folds
avg_accuracy = np.mean(accuracies)
avg_precision = np.mean(precisions)
avg_recall = np.mean(recalls)
avg_f1 = np.mean(f1_scores)
avg_roc_auc = np.mean(roc_aucs)
avg_pr_auc = np.mean(pr_aucs)

# Also compute std for stability
std_accuracy = np.std(accuracies)
std_precision = np.std(precisions)
std_recall = np.std(recalls)
std_f1 = np.std(f1_scores)
std_roc_auc = np.std(roc_aucs)
std_pr_auc = np.std(pr_aucs)

print("\n" + "="*70)
print(f"FINAL RESULTS - {NUM_FOLDS}-Fold Cross-Validation")
print(f"Ablation Mode: '{ablation_mode}'")
print("="*70)
print(f"Accuracy:  {avg_accuracy:.4f} ± {std_accuracy:.4f}")
print(f"Precision: {avg_precision:.4f} ± {std_precision:.4f}")
print(f"Recall:    {avg_recall:.4f} ± {std_recall:.4f}")
print(f"F1 Score:  {avg_f1:.4f} ± {std_f1:.4f}")
print(f"ROC AUC:   {avg_roc_auc:.4f} ± {std_roc_auc:.4f}")
print(f"PR AUC:    {avg_pr_auc:.4f} ± {std_pr_auc:.4f}")
print("="*70)

# Print individual fold results
print("\nIndividual Fold Results:")
print("-"*70)
for i in range(NUM_FOLDS):
    print(f"Fold {i+1}: Acc={accuracies[i]:.4f}, Prec={precisions[i]:.4f}, "
          f"Rec={recalls[i]:.4f}, F1={f1_scores[i]:.4f}, "
          f"ROC-AUC={roc_aucs[i]:.4f}, PR-AUC={pr_aucs[i]:.4f}")
print("-"*70)