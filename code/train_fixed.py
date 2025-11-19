import sys
import torch
import torch.nn as nn
import numpy as np
from model_fixed import AttnFusionGCNNet  # 导入改进后的模型
from utils import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
from torch_geometric.loader import DataLoader
import os
import time
import json

# --- 全局常量 ---
LOG_INTERVAL = 45
NUM_FOLDS = 5
loss_fn = nn.BCELoss()

# ============================================================================
#
# ✨ 超参数配置 (Hyperparameters) - 改进版
#
# ============================================================================
LR = 0.0005
WEIGHT_DECAY = 0.0032
TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
# NUM_EPOCHS = 45
NUM_EPOCHS = 30

# --- 对比学习超参数 ---
ALPHA = 0.3  # miRNA 视图对比损失权重
BETA = 0.3  # Drug 视图对比损失权重
GAMMA = 1.0  # 主任务 (BCE) 权重

WARMUP_EPOCHS = 5  # 对比学习预热轮数
TEMPERATURE = 0.07  # 对比学习温度
LAM = 0.5  # Model_Contrast 内部参数
CONTRASTIVE_DIM = 128

# ============================================================================
# 【新增】融合模块配置
# ============================================================================
USE_IMPROVED_FUSION = True  # True: 双向注意力融合, False: 原版单向注意力
FUSION_TYPE = 'bidirectional'  # 可选: 'bidirectional', 'self_cross', 'co_attention'

# 实验标识
EXPERIMENT_NAME = f"CCL_ASPS_{'Improved' if USE_IMPROVED_FUSION else 'Original'}"

print(f"\n{'=' * 70}")
print(f"🔬 Experiment: {EXPERIMENT_NAME}")
print(f"{'=' * 70}")
print(
    f"[Config] Fusion Strategy: {'Bidirectional Cross-Attention' if USE_IMPROVED_FUSION else 'Original Single-Direction'}")
print(f"[Config] Loss Weights: α(miRNA)={ALPHA}, β(Drug)={BETA}, γ(BCE)={GAMMA}")
print(f"[Config] Total Weighted Loss = {GAMMA}*BCE + warmup_factor*({ALPHA}*miRNA_CL + {BETA}*Drug_CL)")
print(f"[Config] Temperature: {TEMPERATURE}, Warmup Epochs: {WARMUP_EPOCHS}")
print(f"{'=' * 70}\n")


# ============================================================================
#
# Warmup 权重调度
#
# ============================================================================

def get_contrastive_weight(epoch, warmup_epochs=5):
    """
    对比学习权重的平滑 Warmup 策略
    使用余弦 warmup: 0 -> 1
    """
    if epoch <= warmup_epochs:
        progress = epoch / warmup_epochs
        return 0.5 * (1 - np.cos(np.pi * progress))
    return 1.0


# ============================================================================
#
# 核心训练函数 (Train) - 改进版
#
# ============================================================================

def train(model, device, train_loader, optimizer, epoch):
    """
    训练一个 epoch，集成 CCL-ASPS + 改进融合逻辑
    """
    print(f'Training epoch: {epoch}...')
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

        # 前向传播 (支持改进的融合模块)
        output, loss_dict = model(
            data,
            current_epoch=epoch,
            total_epochs=NUM_EPOCHS,
            warmup_epochs=WARMUP_EPOCHS,
            return_contrastive_loss=True
        )

        labels = data.y.view(-1, 1).float().to(device)

        # === 1. BCE 主任务损失 ===
        output = torch.clamp(output, min=1e-7, max=1.0 - 1e-7)
        loss_bce = loss_fn(output, labels)

        # === 2. 对比学习损失 (CCL) ===
        loss_mirna_contrastive = loss_dict['contrastive_mirna']
        loss_drug_contrastive = loss_dict['contrastive_drug']

        # === 3. 总损失融合 ===
        loss = (GAMMA * loss_bce +
                contrastive_weight_factor * (ALPHA * loss_mirna_contrastive +
                                             BETA * loss_drug_contrastive))

        # === 4. 异常检测 ===
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n[!!! 致命错误：损失异常 !!!]")
            print(f"  Epoch: {epoch}, Batch: {batch_idx}")
            print(f"  BCE: {loss_bce.item():.6f}")
            print(f"  miRNA CL: {loss_mirna_contrastive.item():.6f}")
            print(f"  Drug CL: {loss_drug_contrastive.item():.6f}")
            print(f"  Total Loss: {loss.item()}")
            raise ValueError("Loss is NaN/Inf. Stopping training.")

        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

        optimizer.step()

        # === 5. 统计 ===
        total_loss += loss.item()
        total_bce_loss += loss_bce.item()
        total_mirna_contrastive += loss_mirna_contrastive.item()
        total_drug_contrastive += loss_drug_contrastive.item()
        batch_count += 1

        if batch_idx % LOG_INTERVAL == 0:
            print('  Batch: {} [{}/{} ({:.0f}%)]\tTotal: {:.6f} '
                  '(BCE: {:.4f}, miRNA_CL: {:.4f}, Drug_CL: {:.4f}, W: {:.3f})'.format(
                batch_idx,
                batch_idx * train_loader.batch_size,
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item(),
                loss_bce.item(),
                loss_mirna_contrastive.item(),
                loss_drug_contrastive.item(),
                contrastive_weight_factor))

    # === Epoch 总结 ===
    avg_loss = total_loss / batch_count
    avg_bce = total_bce_loss / batch_count
    avg_mirna_cl = total_mirna_contrastive / batch_count
    avg_drug_cl = total_drug_contrastive / batch_count

    print(f'┌─ Epoch {epoch} Summary ─────────────────────────')
    print(f'│ Total Loss:  {avg_loss:.6f}')
    print(f'│ BCE Loss:    {avg_bce:.6f}')
    print(f'│ miRNA CL:    {avg_mirna_cl:.6f}')
    print(f'│ Drug CL:     {avg_drug_cl:.6f}')
    print(f'│ CL Weight:   {contrastive_weight_factor:.3f}')
    print(f'└─────────────────────────────────────────────────')

    return {
        'total_loss': avg_loss,
        'bce_loss': avg_bce,
        'mirna_contrastive': avg_mirna_cl,
        'drug_contrastive': avg_drug_cl,
        'cl_weight': contrastive_weight_factor
    }


# ============================================================================
#
# 预测函数 (Predicting) - 改进版
#
# ============================================================================

def predicting(model, device, loader):
    """
    推理阶段
    """
    model.eval()
    total_probs = []
    total_labels = []

    print('Making prediction for {} samples...'.format(len(loader.dataset)))
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

            probs = output.cpu().numpy()
            total_probs.extend(probs)
            total_labels.extend(data.y.view(-1, 1).cpu().numpy())

    total_probs = np.array(total_probs).flatten()
    total_labels = np.array(total_labels).flatten()
    total_preds = (total_probs >= 0.5).astype(int)

    # 计算指标
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
#
# 【新增】结果保存与对比分析
#
# ============================================================================

def save_experiment_results(metrics_history, config, filename='results_comparison.json'):
    """
    保存实验结果,便于后续对比分析
    """
    results = {
        'experiment_name': EXPERIMENT_NAME,
        'config': config,
        'metrics': {
            'auc_mean': float(np.mean(metrics_history['auc'])),
            'auc_std': float(np.std(metrics_history['auc'])),
            'aupr_mean': float(np.mean(metrics_history['pr_auc'])),
            'aupr_std': float(np.std(metrics_history['pr_auc'])),
            'acc_mean': float(np.mean(metrics_history['acc'])),
            'acc_std': float(np.std(metrics_history['acc'])),
            'f1_mean': float(np.mean(metrics_history['f1'])),
            'f1_std': float(np.std(metrics_history['f1'])),
        },
        'all_folds': {
            'auc': [float(x) for x in metrics_history['auc']],
            'aupr': [float(x) for x in metrics_history['pr_auc']],
            'acc': [float(x) for x in metrics_history['acc']],
            'f1': [float(x) for x in metrics_history['f1']],
        }
    }

    # 如果文件存在,追加结果;否则创建新文件
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = []

    all_results.append(results)

    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[Info] Results saved to {filename}")


# ============================================================================
#
# 主程序 (Main Execution) - 改进版
#
# ============================================================================

if __name__ == "__main__":

    # --- 1. 环境设置 ---
    cuda_name = "cuda:0"
    if len(sys.argv) > 1:
        cuda_name = "cuda:" + str(int(sys.argv[1]))

    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # --- 2. 配置字典 (用于保存) ---
    config = {
        'use_improved_fusion': USE_IMPROVED_FUSION,
        'lr': LR,
        'weight_decay': WEIGHT_DECAY,
        'batch_size': TRAIN_BATCH_SIZE,
        'epochs': NUM_EPOCHS,
        'alpha': ALPHA,
        'beta': BETA,
        'gamma': GAMMA,
        'temperature': TEMPERATURE,
        'warmup_epochs': WARMUP_EPOCHS,
        'lam': LAM
    }

    modeling = AttnFusionGCNNet

    # 结果容器
    metrics_history = {
        'acc': [], 'prec': [], 'rec': [], 'f1': [], 'auc': [], 'pr_auc': []
    }

    # --- 3. 5-Fold CV ---
    for fold in range(NUM_FOLDS):
        print(f"\n{'=' * 70}")
        print(f">>> Fold {fold + 1}/{NUM_FOLDS}")
        print(f"{'=' * 70}")
        fold_start = time.time()

        # 数据加载
        train_data = TestbedDataset(root='data', dataset='train' + str(fold))
        test_data = TestbedDataset(root='data', dataset='test' + str(fold))

        train_loader = DataLoader(
            train_data,
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=True,
            drop_last=True
        )
        test_loader = DataLoader(
            test_data,
            batch_size=TEST_BATCH_SIZE,
            shuffle=False,
            drop_last=False
        )

        print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

        # ============================================================================
        # 【关键】初始化模型 - 支持改进的融合模块
        # ============================================================================
        model = modeling(
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
            lam=LAM,
            use_improved_fusion=USE_IMPROVED_FUSION  # 【新增参数】
        ).to(device)

        # 打印模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model Parameters: {total_params:,} (Trainable: {trainable_params:,})")

        # 优化器
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # 学习率调度
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=NUM_EPOCHS,
            eta_min=LR * 0.01
        )

        # --- 训练循环 ---
        best_auc = 0.0
        best_aupr = 0.0
        patience_counter = 0
        patience_limit = 15  # Early stopping 耐心值

        for epoch in range(1, NUM_EPOCHS + 1):
            train_metrics = train(model, device, train_loader, optimizer, epoch)

            # 更新学习率
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            # 每 3 个 epoch 验证一次
            if epoch % 3 == 0 or epoch == NUM_EPOCHS:
                acc, prec, rec, f1, auc_score, pr_auc_score = predicting(model, device, test_loader)
                print(f"[Validation] Epoch {epoch} | LR: {current_lr:.6f} | "
                      f"AUC={auc_score:.4f}, AUPR={pr_auc_score:.4f}, F1={f1:.4f}")

                # 保存最佳模型
                if auc_score > best_auc:
                    best_auc = auc_score
                    best_aupr = pr_auc_score
                    patience_counter = 0
                    # 可选: 保存模型
                    # torch.save(model.state_dict(), f'best_model_fold{fold}_{EXPERIMENT_NAME}.pth')
                else:
                    patience_counter += 1

                # Early stopping (可选)
                # if patience_counter >= patience_limit:
                #     print(f"Early stopping at epoch {epoch}")
                #     break

        # --- 最终测试 ---
        acc, prec, rec, f1, auc_score, pr_auc_score = predicting(model, device, test_loader)

        metrics_history['acc'].append(acc)
        metrics_history['prec'].append(prec)
        metrics_history['rec'].append(rec)
        metrics_history['f1'].append(f1)
        metrics_history['auc'].append(auc_score)
        metrics_history['pr_auc'].append(pr_auc_score)

        fold_time = time.time() - fold_start
        print(f"\n┌─ Fold {fold + 1} Final Result ─────────────────")
        print(f"│ AUC:       {auc_score:.4f} (Best: {best_auc:.4f})")
        print(f"│ AUPR:      {pr_auc_score:.4f} (Best: {best_aupr:.4f})")
        print(f"│ Accuracy:  {acc:.4f}")
        print(f"│ Precision: {prec:.4f}")
        print(f"│ Recall:    {rec:.4f}")
        print(f"│ F1-Score:  {f1:.4f}")
        print(f"│ Time:      {fold_time:.1f}s")
        print(f"└─────────────────────────────────────────────")

    # --- 4. 最终结果统计 ---
    print("\n" + "=" * 70)
    print(f"FINAL 5-FOLD CV RESULTS - {EXPERIMENT_NAME}")
    print("=" * 70)
    print(f"AUC:       {np.mean(metrics_history['auc']):.4f} ± {np.std(metrics_history['auc']):.4f}")
    print(f"AUPR:      {np.mean(metrics_history['pr_auc']):.4f} ± {np.std(metrics_history['pr_auc']):.4f}")
    print(f"Accuracy:  {np.mean(metrics_history['acc']):.4f} ± {np.std(metrics_history['acc']):.4f}")
    print(f"Precision: {np.mean(metrics_history['prec']):.4f} ± {np.std(metrics_history['prec']):.4f}")
    print(f"Recall:    {np.mean(metrics_history['rec']):.4f} ± {np.std(metrics_history['rec']):.4f}")
    print(f"F1-Score:  {np.mean(metrics_history['f1']):.4f} ± {np.std(metrics_history['f1']):.4f}")
    print("=" * 70)

    # --- 5. 保存结果 ---
    save_experiment_results(metrics_history, config, filename='results_comparison.json')

    print(f"\n✅ Training completed successfully!")
    print(f"💡 To compare with baseline, run again with USE_IMPROVED_FUSION=False")