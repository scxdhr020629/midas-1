import sys
import torch
import torch.nn as nn
import numpy as np
from model_fixed import AttnFusionGCNNet
from utils import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from torch_geometric.loader import DataLoader
import time

# ============================= 配置 =============================
NUM_FOLDS = 5
loss_fn = nn.BCELoss()

LR = 0.0005
WEIGHT_DECAY = 0.0032
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
NUM_EPOCHS = 20          # 固定训练 100 epoch（你可以自行改成 30/50/200 都行）

print("="*70)
print("[Config] Ablation Study — NO Contrastive Learning")
print(f"[Config] Fixed training for {NUM_EPOCHS} epochs (NO early stopping)")
print(f"[Config] Only the model of the LAST epoch will be used for final evaluation")
print("="*70)

# ============================= 训练 / 预测函数 =============================
def train_one_epoch(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)

        output = model(data)                              # (B, 1)
        labels = data.y.view(-1, 1).float().to(device)

        output = torch.clamp(output, min=1e-7, max=1-1e-7)
        loss = loss_fn(output, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(model, device, loader):
    model.eval()
    probs = []
    labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            probs.append(out.cpu().numpy())
            labels.append(data.y.cpu().numpy())

    probs = np.concatenate(probs).flatten()
    labels = np.concatenate(labels).flatten()
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    auc_score = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
    precision_curve, recall_curve, _ = precision_recall_curve(labels, probs)
    aupr = auc(recall_curve, precision_curve)

    return acc, prec, rec, f1, auc_score, aupr


# ============================= 主循环 =============================
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    results = {
        'acc': [], 'prec': [], 'rec': [], 'f1': [], 'auc': [], 'aupr': []
    }

    for fold in range(NUM_FOLDS):
        print("\n" + "="*70)
        print(f"FOLD {fold+1}/{NUM_FOLDS}")
        print("="*70)

        # 数据加载（你原来的划分方式保持不变）
        train_dataset = TestbedDataset(root='data', dataset='train' + str(fold))
        test_dataset  = TestbedDataset(root='data', dataset='test'  + str(fold))

        train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE,
                                  shuffle=True, drop_last=True)
        test_loader  = DataLoader(test_dataset,  batch_size=TEST_BATCH_SIZE,
                                  shuffle=False)

        model = AttnFusionGCNNet(
            n_output=1, n_filters=32, embed_dim=64,
            num_features_xd=78, num_features_smile=66,
            num_features_xt=25, output_dim=128, dropout=0.2
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=NUM_EPOCHS, eta_min=LR*0.01)

        # ------------------- 固定 epoch 训练 -------------------
        for epoch in range(1, NUM_EPOCHS + 1):
            loss = train_one_epoch(model, device, train_loader, optimizer, epoch)
            scheduler.step()

            if epoch % 10 == 0 or epoch == NUM_EPOCHS:
                acc, prec, rec, f1, auc_val, aupr_val = evaluate(model, device, test_loader)
                print(f"Epoch {epoch:03d} | Loss: {loss:.5f} | "
                      f"Test AUC: {auc_val:.4f} | Test AUPR: {aupr_val:.4f}")

        # ------------------- 最终评估：只用最后一轮模型 -------------------
        acc, prec, rec, f1, auc_score, aupr_score = evaluate(model, device, test_loader)

        results['acc'].append(acc)
        results['prec'].append(prec)
        results['rec'].append(rec)
        results['f1'].append(f1)
        results['auc'].append(auc_score)
        results['aupr'].append(aupr_score)

        print(f"\n>>> Fold {fold+1} FINAL (last epoch) → "
              f"AUC: {auc_score:.4f} ± AUPR: {aupr_score:.4f}")

    # ------------------- 5-fold 汇总 -------------------
    print("\n" + "="*70)
    print("5-FOLD CROSS-VALIDATION FINAL RESULTS (last epoch only, no early stopping)")
    print("="*70)
    print(f"AUC      : {np.mean(results['auc']):.4f} ± {np.std(results['auc']):.4f}")
    print(f"AUPR     : {np.mean(results['aupr']):.4f} ± {np.std(results['aupr']):.4f}")
    print(f"Accuracy : {np.mean(results['acc']):.4f} ± {np.std(results['acc']):.4f}")
    print(f"F1       : {np.mean(results['f1']):.4f} ± {np.std(results['f1']):.4f}")
    print("="*70)