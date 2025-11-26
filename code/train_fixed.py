import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model_fixed import AttnFusionGCNNet, Model_Contrast
from utils import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from torch_geometric.loader import DataLoader
import time

# --- 全局常量 ---
LOG_INTERVAL = 45
NUM_FOLDS = 5
loss_fn = nn.BCELoss()

# ============================================================================
# ✨ 超参数配置 - ASPS + InfoNCE
# ============================================================================
LR = 0.0005
WEIGHT_DECAY = 0.001
TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
NUM_EPOCHS = 20

# 对比学习权重
ALPHA = 0.3  # miRNA CL 权重
BETA = 0.3   # Drug CL 权重
GAMMA = 1.0  # BCE 权重

WARMUP_EPOCHS = 5
TEMPERATURE = 0.1

print(f"[Config] ASPS + InfoNCE: α={ALPHA}, β={BETA}, γ={GAMMA}, τ={TEMPERATURE}")


def get_contrastive_weight(epoch, warmup_epochs=5):
    """余弦 Warmup"""
    if epoch <= warmup_epochs:
        import math
        return 0.5 * (1 - math.cos(math.pi * epoch / warmup_epochs))
    return 1.0


# ============================================================================
# 训练函数
# ============================================================================
def train(model, device, train_loader, optimizer, epoch):
    print(f'Training epoch: {epoch}...')
    model.train()

    total_loss, total_bce, total_mirna_cl, total_drug_cl = 0, 0, 0, 0
    batch_count = 0
    cl_weight = get_contrastive_weight(epoch, WARMUP_EPOCHS)

    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)

        try:
            # ⭐ 传递 epoch 参数给模型，用于 ASPS
            output, loss_dict = model(
                data, 
                current_epoch=epoch,
                total_epochs=NUM_EPOCHS,
                warmup_epochs=WARMUP_EPOCHS,
                return_contrastive_loss=True
            )

            labels = data.y.view(-1, 1).float().to(device)
            loss_bce = loss_fn(output, labels)
            loss_mirna = loss_dict['contrastive_mirna']
            loss_drug = loss_dict['contrastive_drug']

            loss = GAMMA * loss_bce + cl_weight * (ALPHA * loss_mirna + BETA * loss_drug)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[WARNING] NaN/Inf at batch {batch_idx}, skipping...")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_bce += loss_bce.item()
            total_mirna_cl += loss_mirna.item()
            total_drug_cl += loss_drug.item()
            batch_count += 1

            if batch_idx % LOG_INTERVAL == 0:
                print(f'  Batch {batch_idx}: Loss={loss.item():.4f} '
                      f'(BCE={loss_bce.item():.4f}, miRNA={loss_mirna.item():.4f}, Drug={loss_drug.item():.4f})')
                      
        except RuntimeError as e:
            print(f"[ERROR] Batch {batch_idx}: {e}")
            continue

    if batch_count == 0:
        return float('inf')
        
    print(f'┌─ Epoch {epoch}: Loss={total_loss/batch_count:.4f}, '
          f'BCE={total_bce/batch_count:.4f}, CL_W={cl_weight:.2f}')
    print(f'└─ miRNA_CL={total_mirna_cl/batch_count:.4f}, Drug_CL={total_drug_cl/batch_count:.4f}')

    return total_loss / batch_count


# ============================================================================
# 预测函数
# ============================================================================
def predicting(model, device, loader):
    """
    推理/验证函数
    
    注意：model.eval() 会：
    1. 关闭 Dropout（不再随机置零）
    2. BatchNorm 使用 running mean/var 而非 batch 统计量
    """
    model.eval()  # ⭐ 关键：确保推理模式
    total_probs, total_labels = [], []

    with torch.no_grad():  # ⭐ 关键：不计算梯度
        for data in loader:
            data = data.to(device)
            output = model(data, return_contrastive_loss=False)
            total_probs.extend(output.cpu().numpy())
            total_labels.extend(data.y.view(-1, 1).cpu().numpy())

    total_probs = np.array(total_probs).flatten()
    total_labels = np.array(total_labels).flatten()
    total_preds = (total_probs >= 0.5).astype(int)

    acc = accuracy_score(total_labels, total_preds)
    prec = precision_score(total_labels, total_preds, zero_division=0)
    rec = recall_score(total_labels, total_preds, zero_division=0)
    f1 = f1_score(total_labels, total_preds, zero_division=0)
    
    try:
        auc_score = roc_auc_score(total_labels, total_probs)
    except:
        auc_score = 0.5
    
    pr_vals, rec_vals, _ = precision_recall_curve(total_labels, total_probs)
    pr_auc = auc(rec_vals, pr_vals)

    return acc, prec, rec, f1, auc_score, pr_auc


# ============================================================================
# 主程序
# ============================================================================
if __name__ == "__main__":
    cuda_name = "cuda:0" if len(sys.argv) <= 1 else f"cuda:{sys.argv[1]}"
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    print(f"纯 SimSiam 对比学习 - 5-Fold CV")
    print("=" * 70)

    metrics_history = {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'auc': [], 'pr_auc': []}

    for fold in range(NUM_FOLDS):
        print(f"\n{'=' * 70}\n>>> Fold {fold + 1}/{NUM_FOLDS}\n{'=' * 70}")
        fold_start = time.time()

        train_data = TestbedDataset(root='data', dataset=f'train{fold}')
        test_data = TestbedDataset(root='data', dataset=f'test{fold}')

        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, drop_last=False)

        print(f"Train: {len(train_data)}, Test: {len(test_data)}")

        model = AttnFusionGCNNet(
            n_output=1, n_filters=32, embed_dim=64,
            num_features_xd=78, num_features_smile=66, num_features_xt=25,
            output_dim=128, dropout=0.2, temperature=TEMPERATURE
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=LR * 0.01)

        for epoch in range(1, NUM_EPOCHS + 1):
            train(model, device, train_loader, optimizer, epoch)
            scheduler.step()

            if epoch % 10 == 0:
                acc, prec, rec, f1, auc_score, pr_auc = predicting(model, device, test_loader)
                print(f"[Val] Epoch {epoch}: AUC={auc_score:.4f}, AUPR={pr_auc:.4f}")

        # Final evaluation
        acc, prec, rec, f1, auc_score, pr_auc = predicting(model, device, test_loader)
        metrics_history['acc'].append(acc)
        metrics_history['prec'].append(prec)
        metrics_history['rec'].append(rec)
        metrics_history['f1'].append(f1)
        metrics_history['auc'].append(auc_score)
        metrics_history['pr_auc'].append(pr_auc)

        print(f"\n┌─ Fold {fold + 1} Result ─────────────────")
        print(f"│ AUC: {auc_score:.4f}, AUPR: {pr_auc:.4f}")
        print(f"│ Acc: {acc:.4f}, F1: {f1:.4f}")
        print(f"│ Time: {time.time() - fold_start:.1f}s")
        print(f"└─────────────────────────────────────────────")

    # Final results
    print("\n" + "=" * 70)
    print("FINAL 5-FOLD CV RESULTS")
    print("=" * 70)
    print(f"AUC:  {np.mean(metrics_history['auc']):.4f} ± {np.std(metrics_history['auc']):.4f}")
    print(f"AUPR: {np.mean(metrics_history['pr_auc']):.4f} ± {np.std(metrics_history['pr_auc']):.4f}")
    print(f"Acc:  {np.mean(metrics_history['acc']):.4f} ± {np.std(metrics_history['acc']):.4f}")
    print(f"F1:   {np.mean(metrics_history['f1']):.4f} ± {np.std(metrics_history['f1']):.4f}")
    print("=" * 70)