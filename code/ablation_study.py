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
import json

# ============================================================================
# 🔬 消融实验配置
# ============================================================================

ABLATION_CONFIGS = {
    # 完整模型（Full Model）
    "full": {
        "use_contrastive": True,
        "use_asps": True,
        "description": "Full Model (Contrastive + ASPS)"
    },
    
    # 无对比学习（No Contrastive Learning）
    "no_contrastive": {
        "use_contrastive": False,
        "use_asps": True,
        "description": "No Contrastive Learning (Only ASPS)"
    },
    
    # 无ASPS（No ASPS - 固定负样本）
    "no_asps": {
        "use_contrastive": True,
        "use_asps": False,
        "description": "No ASPS (Fixed Negative Sampling)"
    },
    
    # 基线模型（Baseline - 无对比学习和ASPS）
    "baseline": {
        "use_contrastive": False,
        "use_asps": False,
        "description": "Baseline (No Contrastive, No ASPS)"
    }
}

# --- 全局常量 ---
LOG_INTERVAL = 45
NUM_FOLDS = 5
loss_fn = nn.BCELoss()

# --- 超参数配置 ---
LR = 0.0005
WEIGHT_DECAY = 0.0032
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
NUM_EPOCHS = 20
WARMUP_EPOCHS = 2

# --- 对比学习参数 ---
ALPHA = 0.5
BETA = 0.5
GAMMA = 1.0
TEMPERATURE = 0.1
LAM = 0.5
CONTRASTIVE_DIM = 128

ASPS_FULL_ACTIVATION_EPOCH = int(WARMUP_EPOCHS + 0.5 * (NUM_EPOCHS - WARMUP_EPOCHS))

# ============================================================================
# 修改的 get_contrast_pair_batch 函数（支持关闭ASPS）
# ============================================================================
def get_contrast_pair_batch_ablation(args, feat_sim, device, use_asps=True):
    """
    支持消融实验的对比样本生成函数
    
    Args:
        use_asps: 如果为False，则使用固定的全负样本（不使用ASPS）
    """
    batch_size = feat_sim.shape[0]
    
    # 基础正样本
    pos = torch.eye(batch_size).to(device)
    
    # 基础负样本
    neg_all = torch.ones_like(pos) - pos
    
    if not use_asps:
        # 关闭ASPS: 直接使用所有负样本
        return pos, neg_all
    
    # 使用ASPS策略
    current_epoch = args.current_epoch if hasattr(args, 'current_epoch') else args['current_epoch']
    total_epoch = args.epochs if hasattr(args, 'epochs') else args['epochs']
    beta = args.beta if hasattr(args, 'beta') else 0.5
    warmup_epochs = args.get('warmup_epochs', 5)
    
    max_neg_num = batch_size - 1
    
    if current_epoch <= warmup_epochs:
        neg = neg_all
    else:
        progress = (current_epoch - warmup_epochs) / (total_epoch - warmup_epochs)
        progress = max(0.0, min(1.0, progress))
        
        k_neg = int(max_neg_num * (progress ** 1.5) * beta)
        k_neg = max(1, min(k_neg, max_neg_num))
        
        feat_sim_masked = feat_sim.clone()
        feat_sim_masked.fill_diagonal_(-9e15)
        
        vals, indices = feat_sim_masked.topk(k=k_neg, dim=1, largest=True)
        
        hard_neg_mask = torch.zeros_like(feat_sim).to(device)
        hard_neg_mask.scatter_(1, indices, 1)
        
        alpha = min(progress * 2, 1.0)
        neg = alpha * hard_neg_mask + (1 - alpha) * neg_all
    
    neg = neg * (1 - pos)
    return pos, neg

# ============================================================================
# 训练函数
# ============================================================================
def get_contrastive_weight(epoch, warmup_epochs=5):
    if epoch <= warmup_epochs:
        progress = epoch / warmup_epochs
        return 0.5 * (1 - np.cos(np.pi * progress))
    return 1.0

def train(model, device, train_loader, optimizer, epoch, config):
    """
    支持消融实验的训练函数
    
    Args:
        config: 消融实验配置字典
    """
    model.train()
    total_loss = 0
    batch_count = 0
    
    use_contrastive = config["use_contrastive"]
    use_asps = config["use_asps"]
    
    contrastive_weight_factor = get_contrastive_weight(epoch, WARMUP_EPOCHS) if use_contrastive else 0.0

    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)

        # 根据配置决定是否返回对比学习损失
        if use_contrastive:
            output, loss_dict = model(
                data,
                current_epoch=epoch,
                total_epochs=NUM_EPOCHS,
                warmup_epochs=WARMUP_EPOCHS,
                return_contrastive_loss=True
            )
        else:
            output = model(
                data,
                current_epoch=epoch,
                total_epochs=NUM_EPOCHS,
                warmup_epochs=WARMUP_EPOCHS,
                return_contrastive_loss=False
            )
            loss_dict = None

        labels = data.y.view(-1, 1).float().to(device)
        output = output.view(-1, 1)
        output = torch.clamp(output, min=1e-7, max=1.0 - 1e-7)
        
        loss_bce = loss_fn(output, labels)
        
        # 计算总损失
        if use_contrastive and loss_dict is not None:
            loss_mirna_contrastive = loss_dict['contrastive_mirna']
            loss_drug_contrastive = loss_dict['contrastive_drug']
            
            loss = (GAMMA * loss_bce +
                    contrastive_weight_factor * (ALPHA * loss_mirna_contrastive +
                                                 BETA * loss_drug_contrastive))
        else:
            # 只使用BCE损失
            loss = loss_bce

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
# 运行单个消融实验
# ============================================================================
def run_ablation_experiment(config_name, config, device):
    """运行单个消融实验配置"""
    
    print(f"\n{'='*80}")
    print(f"🔬 Ablation Experiment: {config_name.upper()}")
    print(f"📋 Configuration: {config['description']}")
    print(f"   - Use Contrastive Learning: {config['use_contrastive']}")
    print(f"   - Use ASPS: {config['use_asps']}")
    print(f"{'='*80}\n")
    
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

        # 训练循环
        for epoch in range(1, NUM_EPOCHS + 1):
            loss_val = train(model, device, train_loader, optimizer, epoch, config)
            if loss_val is None:
                break

            scheduler.step()
            acc, prec, rec, f1, auc_score, pr_auc_score = predicting(model, device, test_loader)
            
            # 显示训练阶段
            if epoch <= WARMUP_EPOCHS:
                phase = "Warmup"
            elif config['use_asps'] and epoch < ASPS_FULL_ACTIVATION_EPOCH:
                phase = f"ASPS Ramping"
            else:
                phase = "Training"
            
            if epoch % 5 == 0 or epoch == NUM_EPOCHS:
                print(f'Epoch {epoch:03d} [{phase}]: Loss={loss_val:.5f} | Val AUC={auc_score:.5f}')

        # 最终评估
        print(f"\n使用最后一轮 (Epoch {NUM_EPOCHS}) 的模型进行最终评估...")
        acc, prec, rec, f1, auc_score, pr_auc_score = predicting(model, device, test_loader)

        metrics_history['acc'].append(acc)
        metrics_history['prec'].append(prec)
        metrics_history['rec'].append(rec)
        metrics_history['f1'].append(f1)
        metrics_history['auc'].append(auc_score)
        metrics_history['pr_auc'].append(pr_auc_score)

        print(f"Fold {fold + 1} Final Result → AUC: {auc_score:.4f}, AUPR: {pr_auc_score:.4f}")

    return metrics_history

# ============================================================================
# 主程序 - 运行所有消融实验
# ============================================================================
if __name__ == "__main__":
    cuda_name = "cuda:0"
    if len(sys.argv) > 1:
        cuda_name = "cuda:" + str(int(sys.argv[1]))

    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}\n')
    
    # 存储所有实验结果
    all_results = {}
    
    # 运行所有消融实验
    for config_name, config in ABLATION_CONFIGS.items():
        results = run_ablation_experiment(config_name, config, device)
        all_results[config_name] = results
    
    # ============================================================================
    # 📊 打印汇总结果对比表
    # ============================================================================
    print("\n" + "="*100)
    print("📊 ABLATION STUDY RESULTS SUMMARY")
    print("="*100)
    
    print(f"\n{'Configuration':<30} {'AUC':<20} {'AUPR':<20} {'Accuracy':<20}")
    print("-"*100)
    
    for config_name, config in ABLATION_CONFIGS.items():
        results = all_results[config_name]
        auc_mean = np.mean(results['auc'])
        auc_std = np.std(results['auc'])
        aupr_mean = np.mean(results['pr_auc'])
        aupr_std = np.std(results['pr_auc'])
        acc_mean = np.mean(results['acc'])
        acc_std = np.std(results['acc'])
        
        print(f"{config['description']:<30} "
              f"{auc_mean:.4f}±{auc_std:.4f}    "
              f"{aupr_mean:.4f}±{aupr_std:.4f}    "
              f"{acc_mean:.4f}±{acc_std:.4f}")
    
    print("="*100)
    
    # ============================================================================
    # 📈 计算性能提升百分比
    # ============================================================================
    print("\n" + "="*100)
    print("📈 PERFORMANCE IMPROVEMENT ANALYSIS (vs Baseline)")
    print("="*100)
    
    baseline_auc = np.mean(all_results['baseline']['auc'])
    baseline_aupr = np.mean(all_results['baseline']['pr_auc'])
    
    print(f"\n{'Configuration':<30} {'AUC Improvement':<25} {'AUPR Improvement':<25}")
    print("-"*100)
    
    for config_name, config in ABLATION_CONFIGS.items():
        if config_name == 'baseline':
            continue
        
        results = all_results[config_name]
        auc_mean = np.mean(results['auc'])
        aupr_mean = np.mean(results['pr_auc'])
        
        auc_improvement = ((auc_mean - baseline_auc) / baseline_auc) * 100
        aupr_improvement = ((aupr_mean - baseline_aupr) / baseline_aupr) * 100
        
        print(f"{config['description']:<30} "
              f"{auc_improvement:+.2f}%                 "
              f"{aupr_improvement:+.2f}%")
    
    print("="*100)
    
    # ============================================================================
    # 💾 保存结果到JSON文件
    # ============================================================================
    results_summary = {}
    for config_name, config in ABLATION_CONFIGS.items():
        results = all_results[config_name]
        results_summary[config_name] = {
            'description': config['description'],
            'use_contrastive': config['use_contrastive'],
            'use_asps': config['use_asps'],
            'metrics': {
                'auc': {
                    'mean': float(np.mean(results['auc'])),
                    'std': float(np.std(results['auc'])),
                    'values': [float(x) for x in results['auc']]
                },
                'aupr': {
                    'mean': float(np.mean(results['pr_auc'])),
                    'std': float(np.std(results['pr_auc'])),
                    'values': [float(x) for x in results['pr_auc']]
                },
                'accuracy': {
                    'mean': float(np.mean(results['acc'])),
                    'std': float(np.std(results['acc'])),
                    'values': [float(x) for x in results['acc']]
                }
            }
        }
    
    with open('ablation_results.json', 'w') as f:
        json.dump(results_summary, f, indent=4)
    
    print(f"\n✅ Results saved to 'ablation_results.json'")
    print("\n" + "="*100)