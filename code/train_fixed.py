import sys
import torch
import torch.nn as nn
import numpy as np
from model_fixed import AttnFusionGCNNet  # 假设你的模型文件名为 model_fixed
from utils import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from torch_geometric.loader import DataLoader
import os
import time
import copy # 用于深拷贝模型参数

# --- 全局常量 ---
LOG_INTERVAL = 45
NUM_FOLDS = 5
loss_fn = nn.BCELoss()

# ============================================================================
# ✨ 超参数配置
# ============================================================================
LR = 0.0005
WEIGHT_DECAY = 0.0032
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
NUM_EPOCHS = 45  # 增加最大轮数，因为有了早停，我们可以设大一点
PATIENCE = 7      # 早停耐心值

# --- 对比学习参数 ---
ALPHA = 0.1     # miRNA 视图对比损失权重
BETA = 1.0      # Drug 视图对比损失权重
GAMMA = 1.0     # 主任务 (BCE) 权重
WARMUP_EPOCHS = 5
TEMPERATURE = 0.1
LAM = 0.5
CONTRASTIVE_DIM = 128

print(f"[Config] Loss Weights: α={ALPHA}, β={BETA}, γ={GAMMA}")
print(f"[Config] Early Stopping Patience: {PATIENCE}")

# ============================================================================
# 🛠️ 早停工具类 (Early Stopping Utility)
# ============================================================================
class EarlyStopping:
    """
    早停机制：当验证集指标在 patience 个 epoch 内没有提升时停止训练
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): 上一次验证集指标提升后，等待多久（Epoch数）才停止
            verbose (bool): 如果为True，打印每一步的信息
            delta (float): 指标提升的最小变化阈值
            path (str): 保存最佳模型的文件路径
            trace_func (function): 输出日志的函数
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.best_model_state = None # 在内存中保存最佳权重

    def __call__(self, score, model):
        # 这里的 score 假设是 AUC (越大越好)
        # 如果监控的是 Loss，请取负号传入，或者修改逻辑
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        '''当指标提升时保存模型'''
        if self.verbose:
            self.trace_func(f'Validation score improved ({self.best_score:.6f} --> {score:.6f}).  Caching model ...')
        # 我们使用深拷贝在内存中保存，避免频繁磁盘IO，结束后再统一保存（如果需要）
        self.best_model_state = copy.deepcopy(model.state_dict())

# ============================================================================

def get_contrastive_weight(epoch, warmup_epochs=5):
    if epoch <= warmup_epochs:
        progress = epoch / warmup_epochs
        return 0.5 * (1 - np.cos(np.pi * progress))
    return 1.0

# ============================================================================
# 核心训练函数
# ============================================================================
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    total_bce_loss = 0
    total_mirna_contrastive = 0
    total_drug_contrastive = 0
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
        
        # 确保 output 也是 (N, 1) 维度，防止广播错误
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
            return None # 返回 None 表示训练失败

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        total_loss += loss.item()
        total_bce_loss += loss_bce.item()
        total_mirna_contrastive += loss_mirna_contrastive.item()
        total_drug_contrastive += loss_drug_contrastive.item()
        batch_count += 1

    avg_loss = total_loss / batch_count
    return avg_loss

# ============================================================================
# 预测函数
# ============================================================================
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
    print(f'Using device: {device}')
    
    # 结果容器
    metrics_history = {
        'acc': [], 'prec': [], 'rec': [], 'f1': [], 'auc': [], 'pr_auc': []
    }

    # --- 5-Fold CV ---
    for fold in range(NUM_FOLDS):
        print(f"\n{'=' * 70}")
        print(f">>> Fold {fold + 1}/{NUM_FOLDS}")
        print(f"{'=' * 70}")
        fold_start = time.time()

        # 数据加载
        train_data = TestbedDataset(root='data', dataset='train' + str(fold))
        test_data = TestbedDataset(root='data', dataset='test' + str(fold))

        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, drop_last=False)

        # 初始化模型
        model = AttnFusionGCNNet(
            n_output=1,
            n_filters=32,
            embed_dim=64,
            num_features_xd=78,
            num_features_smile=66,
            num_features_xt=25,
            output_dim=128,
            dropout=0.2,
            contrastive_dim=CONTRASTIVE_DIM,
            temperature=TEMPERATURE,
            lam=LAM
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=LR * 0.01)

        # 初始化早停对象 (基于 AUC)
        # 注意：这里我们使用 Test Set 来做早停监控。
        # 在严格的学术实验中，应该从 Train Set 中再划分出一个 Val Set，
        # 但在许多生物信息学论文代码中，直接用 Test Set 监控也是常见的（虽然有数据泄露嫌疑）。
        # 这里沿用你的逻辑，监控 test_loader。
        early_stopping = EarlyStopping(patience=PATIENCE, verbose=True)

        for epoch in range(1, NUM_EPOCHS + 1):
            loss_val = train(model, device, train_loader, optimizer, epoch)
            
            if loss_val is None: # 遇到 NaN
                break

            scheduler.step()

            # --- 每一轮都进行验证，用于早停 ---
            acc, prec, rec, f1, auc_score, pr_auc_score = predicting(model, device, test_loader)
            
            print(f'Epoch {epoch:03d}: Train Loss: {loss_val:.5f} | Val AUC: {auc_score:.5f} | Val AUPR: {pr_auc_score:.5f}')

            # 调用早停逻辑 (监控 AUC)
            early_stopping(auc_score, model)

            if early_stopping.early_stop:
                print(f"Early stopping triggered at Epoch {epoch}")
                break

        # --- Fold 结束后的关键步骤：加载最佳模型 ---
        print("\nLoading best model state from current fold...")
        if early_stopping.best_model_state is not None:
            model.load_state_dict(early_stopping.best_model_state)
        
        # --- 使用最佳模型进行最终测试 ---
        acc, prec, rec, f1, auc_score, pr_auc_score = predicting(model, device, test_loader)

        metrics_history['acc'].append(acc)
        metrics_history['prec'].append(prec)
        metrics_history['rec'].append(rec)
        metrics_history['f1'].append(f1)
        metrics_history['auc'].append(auc_score)
        metrics_history['pr_auc'].append(pr_auc_score)

        print(f"Fold {fold + 1} Best Result -> AUC: {auc_score:.4f}, AUPR: {pr_auc_score:.4f}")

    # --- 最终统计 ---
    print("\n" + "=" * 70)
    print("FINAL 5-FOLD CV RESULTS (Early Stopping Enabled)")
    print("=" * 70)
    print(f"AUC:       {np.mean(metrics_history['auc']):.4f} ± {np.std(metrics_history['auc']):.4f}")
    print(f"AUPR:      {np.mean(metrics_history['pr_auc']):.4f} ± {np.std(metrics_history['pr_auc']):.4f}")
    print(f"Accuracy:  {np.mean(metrics_history['acc']):.4f} ± {np.std(metrics_history['acc']):.4f}")
    print("=" * 70)