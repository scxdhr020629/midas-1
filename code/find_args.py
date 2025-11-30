import sys
import torch
import torch.nn as nn
import numpy as np
import random
import json
import time
import itertools
from model_fixed import AttnFusionGCNNet  # 确保 model_fixed.py 在同一目录下
from utils import TestbedDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from torch_geometric.loader import DataLoader

# ============================================================================
# 🎛️ 调参搜索空间配置 (Hyperparameter Search Space)
# ============================================================================

# 搜索尝试次数 (根据你的显卡时间和耐心设置，建议 20-50 次)
N_TRIALS = 20  

# 定义参数网格
PARAM_GRID = {
    'LR': [0.0001, 0.0005, 0.001],
    'WEIGHT_DECAY': [1e-4, 1e-3, 0.0032],
    'BATCH_SIZE': [64, 128, 256],           # 显存允许的话尽量大
    'ALPHA': [0.1, 0.3, 0.5, 1.0],          # miRNA CL 权重
    'BETA': [0.1, 0.3, 0.5, 1.0],           # Drug CL 权重
    'GAMMA': [0.5, 1.0, 2.0],               # BCE 主任务权重
    'TEMPERATURE': [0.05, 0.07, 0.1, 0.2],  # 对比学习温度
    'DROPOUT': [0.2, 0.3, 0.4]              # 防止过拟合
}

# 固定配置
NUM_FOLDS = 5
MAX_EPOCHS = 50       # 设置一个较大的上限，依靠早停来停止
PATIENCE = 7          # 早停耐心值
WARMUP_EPOCHS = 5
LOG_FILE = "tuning_log.txt"
BEST_PARAMS_FILE = "best_params_final.json"

# ============================================================================
# 工具函数
# ============================================================================

def get_contrastive_weight(epoch, warmup_epochs=5):
    """余弦预热策略"""
    if epoch <= warmup_epochs:
        progress = epoch / warmup_epochs
        return 0.5 * (1 - np.cos(np.pi * progress))
    return 1.0

def log_to_file(message):
    """写入日志文件并打印"""
    print(message)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")

def get_random_params(grid):
    """从网格中随机采样一组参数"""
    params = {}
    for key, values in grid.items():
        params[key] = random.choice(values)
    return params

# ============================================================================
# 训练与验证逻辑
# ============================================================================

def train_one_epoch(model, device, train_loader, optimizer, epoch, config):
    model.train()
    loss_fn = nn.BCELoss()
    
    contrastive_weight_factor = get_contrastive_weight(epoch, WARMUP_EPOCHS)
    
    total_loss = 0
    
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        
        output, loss_dict = model(
            data,
            current_epoch=epoch,
            total_epochs=MAX_EPOCHS,
            warmup_epochs=WARMUP_EPOCHS,
            return_contrastive_loss=True
        )
        
        labels = data.y.view(-1, 1).float().to(device)
        
        # 1. BCE Loss
        output = torch.clamp(output, min=1e-7, max=1.0 - 1e-7)
        loss_bce = loss_fn(output, labels)
        
        # 2. Contrastive Loss
        loss_mirna = loss_dict['contrastive_mirna']
        loss_drug = loss_dict['contrastive_drug']
        
        # 3. Weighted Sum
        loss = (config['GAMMA'] * loss_bce + 
                contrastive_weight_factor * (config['ALPHA'] * loss_mirna + 
                                           config['BETA'] * loss_drug))
        
        if torch.isnan(loss) or torch.isinf(loss):
            return float('nan') # 抛出 NaN 信号
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def evaluate(model, device, loader):
    model.eval()
    total_probs = []
    total_labels = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data, current_epoch=0, total_epochs=MAX_EPOCHS, 
                         warmup_epochs=WARMUP_EPOCHS, return_contrastive_loss=False)
            total_probs.extend(output.cpu().numpy())
            total_labels.extend(data.y.view(-1, 1).cpu().numpy())
            
    total_probs = np.array(total_probs).flatten()
    total_labels = np.array(total_labels).flatten()
    
    try:
        auc_score = roc_auc_score(total_labels, total_probs)
    except:
        auc_score = 0.5
        
    return auc_score

# ============================================================================
# 单次交叉验证流程 (Run one set of hyperparameters)
# ============================================================================

def run_cv_trial(trial_idx, config, device):
    log_to_file(f"\n{'='*60}")
    log_to_file(f"🚀 Trial {trial_idx+1}/{N_TRIALS} | Params: {json.dumps(config)}")
    log_to_file(f"{'='*60}")
    
    fold_aucs = []
    
    # 开始 5-Fold CV
    for fold in range(NUM_FOLDS):
        # print(f"  Running Fold {fold+1}/{NUM_FOLDS}...")
        
        # 数据加载 (使用 Config 中的 Batch Size)
        train_data = TestbedDataset(root='data', dataset='train' + str(fold))
        test_data = TestbedDataset(root='data', dataset='test' + str(fold))
        
        train_loader = DataLoader(train_data, batch_size=config['BATCH_SIZE'], shuffle=True, drop_last=True)
        test_loader = DataLoader(test_data, batch_size=config['BATCH_SIZE'], shuffle=False, drop_last=False)
        
        # 模型初始化 (使用 Config 中的参数)
        model = AttnFusionGCNNet(
            n_output=1,
            n_filters=32,
            embed_dim=64,
            num_features_xd=78,
            num_features_smile=66,
            num_features_xt=25,
            output_dim=128,
            dropout=config['DROPOUT'],       # 动态 Dropout
            contrastive_dim=128,
            temperature=config['TEMPERATURE'], # 动态 Temperature
            lam=0.5
        ).to(device)
        
        # 优化器 (使用 Config 中的 LR 和 Weight Decay)
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config['LR'], 
            weight_decay=config['WEIGHT_DECAY']
        )
        
        # 学习率调度
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MAX_EPOCHS, eta_min=config['LR'] * 0.01
        )
        
        # 内部训练循环 (带早停)
        best_fold_auc = 0.0
        patience_counter = 0
        
        for epoch in range(1, MAX_EPOCHS + 1):
            train_loss = train_one_epoch(model, device, train_loader, optimizer, epoch, config)
            
            # 检测 Loss NaN
            if np.isnan(train_loss):
                log_to_file(f"  ❌ Fold {fold+1} Failed: Loss became NaN at epoch {epoch}")
                return float('nan') # 整个 Trial 失败
            
            scheduler.step()
            
            # 验证
            val_auc = evaluate(model, device, test_loader)
            
            # 早停逻辑
            if val_auc > best_fold_auc:
                best_fold_auc = val_auc
                patience_counter = 0
                # 这里可以保存 Fold 最佳模型，如果需要
            else:
                patience_counter += 1
                
            if patience_counter >= PATIENCE:
                # print(f"    Early stopping at epoch {epoch} (Best AUC: {best_fold_auc:.4f})")
                break
        
        fold_aucs.append(best_fold_auc)
        print(f"  Fold {fold+1} Finished. Best AUC: {best_fold_auc:.4f}")

    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    
    log_to_file(f"✅ Trial {trial_idx+1} Result: Mean AUC = {mean_auc:.4f} ± {std_auc:.4f}")
    return mean_auc

# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    # 初始化
    cuda_name = "cuda:0"
    if len(sys.argv) > 1:
        cuda_name = "cuda:" + str(int(sys.argv[1]))
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    
    print(f"Starting Hyperparameter Tuning on {device}...")
    print(f"Logs will be saved to {LOG_FILE}")
    
    # 清空之前的日志
    with open(LOG_FILE, "w") as f:
        f.write("Hyperparameter Tuning Start\n")
    
    overall_best_auc = 0.0
    overall_best_params = None
    
    try:
        for i in range(N_TRIALS):
            # 1. 随机采样参数
            current_params = get_random_params(PARAM_GRID)
            
            # 2. 运行 CV
            try:
                mean_auc = run_cv_trial(i, current_params, device)
            except Exception as e:
                log_to_file(f"❌ Trial {i+1} Exception: {str(e)}")
                mean_auc = float('nan')
            
            # 3. 记录并比较结果
            if not np.isnan(mean_auc):
                if mean_auc > overall_best_auc:
                    overall_best_auc = mean_auc
                    overall_best_params = current_params
                    
                    log_to_file(f"🎉 New Best Found! AUC: {overall_best_auc:.4f}")
                    
                    # 立即保存当前最佳参数，防止程序中断丢失
                    with open(BEST_PARAMS_FILE, "w") as f:
                        json.dump(overall_best_params, f, indent=4)
                    
            # 显存清理
            torch.cuda.empty_cache()
            
    except KeyboardInterrupt:
        print("\nTuning interrupted by user.")
        
    print("\n" + "="*60)
    print("🏁 Tuning Completed!")
    print(f"🏆 Overall Best AUC: {overall_best_auc:.4f}")
    print(f"💾 Best Params saved to: {BEST_PARAMS_FILE}")
    print("="*60)
    
    if overall_best_params:
        print(json.dumps(overall_best_params, indent=4))