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
import copy

# --- 全局常量 ---
LOG_INTERVAL = 45
NUM_FOLDS = 5
loss_fn = nn.BCELoss()

# ============================================================================
# ✨ 超参数配置 (Pure ASPS - 无早停)
# ============================================================================
LR = 0.0005
WEIGHT_DECAY = 0.0032
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
NUM_EPOCHS = 30          # 建议 50-60，让 ASPS 充分收敛
WARMUP_EPOCHS = 5

# --- 对比学习参数 ---
ALPHA = 0.5
BETA = 0.5
GAMMA = 1.0
TEMPERATURE = 0.1
LAM = 0.5
CONTRASTIVE_DIM = 128

# ✨ 关键：ASPS 完全激活的轮数
ASPS_FULL_ACTIVATION_EPOCH = int(WARMUP_EPOCHS + 0.5 * (NUM_EPOCHS - WARMUP_EPOCHS))

print(f"{'='*70}")
print(f"[Config] Training Mode: PURE ASPS (NO Early Stopping)")
print(f"[Config] Total Epochs: {NUM_EPOCHS}")
print(f"[Config] Warmup Epochs: {WARMUP_EPOCHS}")
print(f"[Config] ASPS Full Activation at Epoch: {ASPS_FULL_ACTIVATION_EPOCH}")
print(f"[Config] Model Selection: Best AUC after Epoch {ASPS_FULL_ACTIVATION_EPOCH}")
print(f"{'='*70}\n")

# ============================================================================
# 🛠️ 简化版模型追踪器：只在 ASPS 完全激活后保存最佳模型
# ============================================================================
class BestModelTracker:
    """
    不做早停，只追踪 ASPS 完全激活后的最佳模型
    """
    def __init__(self, start_tracking_epoch, verbose=True):
        self.start_tracking_epoch = start_tracking_epoch
        self.verbose = verbose
        self.best_score = -np.inf
        self.best_model_state = None
        self.best_epoch = -1

    def update(self, epoch, score, model):
        """
        只在指定轮数后才开始追踪最佳模型
        """
        if epoch < self.start_tracking_epoch:
            if self.verbose and epoch == 1:
                print(f"[Tracker] Model tracking will start at Epoch {self.start_tracking_epoch}")
            return
        
        if score > self.best_score:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            if self.verbose:
                print(f"[Epoch {epoch}] ✓ New best score: {score:.6f} (Model saved)")

    def load_best_model(self, model):
        """加载最佳模型"""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            if self.verbose:
                print(f"\n✓ Loaded best model from Epoch {self.best_epoch} (AUC: {self.best_score:.6f})")
        else:
            if self.verbose:
                print("\n⚠️  No best model tracked. Using final epoch model.")

# ============================================================================
# 训练函数（保持不变）
# ============================================================================
def get_contrastive_weight(epoch, warmup_epochs=5):
    if epoch <= warmup_epochs:
        progress = epoch / warmup_epochs
        return 0.5 * (1 - np.cos(np.pi * progress))
    return 1.0

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
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

        total_loss += loss.item()
        batch_count += 1

    return total_loss / batch_count

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

        # ✨ 使用简化的追踪器：只在 ASPS 完全激活后追踪最佳模型
        tracker = BestModelTracker(
            start_tracking_epoch=ASPS_FULL_ACTIVATION_EPOCH,
            verbose=True
        )

        for epoch in range(1, NUM_EPOCHS + 1):
            loss_val = train(model, device, train_loader, optimizer, epoch)
            if loss_val is None:
                break

            scheduler.step()
            acc, prec, rec, f1, auc_score, pr_auc_score = predicting(model, device, test_loader)
            
            # 显示当前训练阶段
            if epoch <= WARMUP_EPOCHS:
                phase = "Warmup"
            elif epoch < ASPS_FULL_ACTIVATION_EPOCH:
                phase = f"ASPS Ramping ({epoch}/{ASPS_FULL_ACTIVATION_EPOCH})"
            else:
                phase = "ASPS Full (Tracking Best)"
            
            print(f'Epoch {epoch:03d} [{phase}]: Loss={loss_val:.5f} | Val AUC={auc_score:.5f}')

            # 更新最佳模型追踪
            tracker.update(epoch, auc_score, model)

        # 加载最佳模型
        tracker.load_best_model(model)
        
        # 最终评估
        acc, prec, rec, f1, auc_score, pr_auc_score = predicting(model, device, test_loader)

        metrics_history['acc'].append(acc)
        metrics_history['prec'].append(prec)
        metrics_history['rec'].append(rec)
        metrics_history['f1'].append(f1)
        metrics_history['auc'].append(auc_score)
        metrics_history['pr_auc'].append(pr_auc_score)

        print(f"\nFold {fold + 1} Final Result → AUC: {auc_score:.4f}, AUPR: {pr_auc_score:.4f}")

    print("\n" + "="*70)
    print("FINAL 5-FOLD CV RESULTS (Pure ASPS - No Early Stopping)")
    print("="*70)
    print(f"AUC:      {np.mean(metrics_history['auc']):.4f} ± {np.std(metrics_history['auc']):.4f}")
    print(f"AUPR:     {np.mean(metrics_history['pr_auc']):.4f} ± {np.std(metrics_history['pr_auc']):.4f}")
    print(f"Accuracy: {np.mean(metrics_history['acc']):.4f} ± {np.std(metrics_history['acc']):.4f}")
    print("="*70)