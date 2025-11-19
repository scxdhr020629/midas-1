import sys
import torch
import torch.nn as nn
import numpy as np
from model_fixed import AttnFusionGCNNet  # 修复: 导入修复后的模型
from utils import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
from torch_geometric.loader import DataLoader
import os
import time

# --- 全局常量 ---
LOG_INTERVAL = 45
NUM_FOLDS = 5
loss_fn = nn.BCELoss()

# ============================================================================
#
# ✨ 超参数配置 (Hyperparameters) - 修复版
#
# ============================================================================
LR = 0.0005
WEIGHT_DECAY = 0.0032
TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
NUM_EPOCHS = 45

# --- 对比学习超参数 ---
# 修复: 不再归一化权重，直接使用原始值
# 这样更直观：如果想增强某个损失，直接调大对应权重即可
ALPHA = 0.3  # miRNA 视图对比损失权重
BETA = 0.3  # Drug 视图对比损失权重
GAMMA = 1.0  # 主任务 (BCE) 权重 - 修复: 增加 BCE 权重以稳定训练

WARMUP_EPOCHS = 5  # 对比学习预热轮数
TEMPERATURE = 0.07  # 修复: 降低温度以增强对比 (0.07-0.1 较常用)
LAM = 0.5  # Model_Contrast 内部参数
CONTRASTIVE_DIM = 128

# 修复: 移除权重归一化逻辑
# 用户可以根据实际效果自由调整各权重的相对大小

print(f"[Config] Loss Weights: α(miRNA)={ALPHA}, β(Drug)={BETA}, γ(BCE)={GAMMA}")
print(f"[Config] Total Weighted Loss = {GAMMA}*BCE + warmup_factor*({ALPHA}*miRNA_CL + {BETA}*Drug_CL)")


# ============================================================================


def get_contrastive_weight(epoch, warmup_epochs=5):
    """
    修复: 更平滑的 Warmup 策略

    使用余弦 warmup 而非线性，避免突变
    """
    if epoch <= warmup_epochs:
        # 余弦 warmup: 0 -> 1
        progress = epoch / warmup_epochs
        return 0.5 * (1 - np.cos(np.pi * progress))  # 从 0 平滑增长到 1
    return 1.0


# ============================================================================
#
# 核心训练函数 (Train) - 修复版
#
# ============================================================================

def train(model, device, train_loader, optimizer, epoch):
    """
    训练一个 epoch，集成 CCL-ASPS 逻辑 (修复版)
    """
    print(f'Training epoch: {epoch}...')
    model.train()

    total_loss = 0
    total_bce_loss = 0
    total_mirna_contrastive = 0
    total_drug_contrastive = 0
    batch_count = 0

    # 修复: 使用新的平滑 warmup 策略
    contrastive_weight_factor = get_contrastive_weight(epoch, WARMUP_EPOCHS)

    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)

        # 修复: 传入 warmup_epochs 参数
        output, loss_dict = model(
            data,
            current_epoch=epoch,
            total_epochs=NUM_EPOCHS,
            warmup_epochs=WARMUP_EPOCHS,  # 新增参数
            return_contrastive_loss=True
        )

        labels = data.y.view(-1, 1).float().to(device)

        # === 1. BCE 主任务损失 ===
        output = torch.clamp(output, min=1e-7, max=1.0 - 1e-7)
        loss_bce = loss_fn(output, labels)

        # === 2. 对比学习损失 (CCL) ===
        loss_mirna_contrastive = loss_dict['contrastive_mirna']
        loss_drug_contrastive = loss_dict['contrastive_drug']

        # === 3. 总损失融合 (修复版) ===
        # 修复: 不再归一化权重，直接使用配置值
        loss = (GAMMA * loss_bce +
                contrastive_weight_factor * (ALPHA * loss_mirna_contrastive +
                                             BETA * loss_drug_contrastive))

        # === 4. 异常检测 (增强版) ===
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n[!!! 致命错误：损失异常 !!!]")
            print(f"  Epoch: {epoch}, Batch: {batch_idx}")
            print(f"  BCE: {loss_bce.item():.6f}")
            print(f"  miRNA CL: {loss_mirna_contrastive.item():.6f}")
            print(f"  Drug CL: {loss_drug_contrastive.item():.6f}")
            print(f"  Total Loss: {loss.item()}")

            # 调试: 打印模型参数统计
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"  {name}: grad_norm={param.grad.norm().item():.4f}")

            raise ValueError("Loss is NaN/Inf. Stopping training.")

        loss.backward()

        # 修复: 更严格的梯度裁剪
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
# 预测函数 (Predicting) - 修复版
#
# ============================================================================

def predicting(model, device, loader):
    """
    推理阶段 (修复版)
    """
    model.eval()
    total_probs = []
    total_labels = []

    print('Making prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            # 修复: 推理时明确传入参数，避免使用默认值
            output = model(
                data,
                current_epoch=0,  # 推理时 epoch 无关紧要
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
# 主程序 (Main Execution) - 修复版
#
# ============================================================================

if __name__ == "__main__":

    # --- 1. 环境设置 ---
    cuda_name = "cuda:0"
    if len(sys.argv) > 1:
        cuda_name = "cuda:" + str(int(sys.argv[1]))

    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    print(f"Running 5-Fold CV with CCL-ASPS Model (Fixed Version)...")
    print("=" * 70)

    modeling = AttnFusionGCNNet

    # 打印配置
    print(f"Configuration:")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch Size (Train/Test): {TRAIN_BATCH_SIZE}/{TEST_BATCH_SIZE}")
    print(f"  Loss Weights: α={ALPHA}, β={BETA}, γ={GAMMA}")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  Warmup Epochs: {WARMUP_EPOCHS}")
    print(f"  Lambda: {LAM}")
    print("=" * 70)

    # 结果容器
    metrics_history = {
        'acc': [], 'prec': [], 'rec': [], 'f1': [], 'auc': [], 'pr_auc': []
    }

    # --- 2. 5-Fold CV ---
    for fold in range(NUM_FOLDS):
        print(f"\n{'=' * 70}")
        print(f">>> Fold {fold + 1}/{NUM_FOLDS}")
        print(f"{'=' * 70}")
        fold_start = time.time()

        # 数据加载
        train_data = TestbedDataset(root='data', dataset='train' + str(fold))
        test_data = TestbedDataset(root='data', dataset='test' + str(fold))

        # 修复: 测试集不使用 drop_last，避免丢失数据
        train_loader = DataLoader(
            train_data,
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=True,
            drop_last=True  # 训练时 drop_last 确保批次一致
        )
        test_loader = DataLoader(
            test_data,
            batch_size=TEST_BATCH_SIZE,
            shuffle=False,
            drop_last=False  # 修复: 测试时保留所有样本
        )

        print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

        # 初始化模型
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
            lam=LAM
        ).to(device)

        # 修复: 使用 AdamW 优化器 (更稳定)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # 修复: 更温和的学习率调度
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=NUM_EPOCHS,
            eta_min=LR * 0.01
        )

        # --- 训练循环 ---
        best_auc = 0.0
        patience_counter = 0
        patience_limit = 10

        for epoch in range(1, NUM_EPOCHS + 1):
            train_metrics = train(model, device, train_loader, optimizer, epoch)

            # 更新学习率
            scheduler.step()

            # 修复: 每 5 个 epoch 进行一次验证（可选）
            if epoch % 5 == 0 or epoch == NUM_EPOCHS:
                acc, prec, rec, f1, auc_score, pr_auc_score = predicting(model, device, test_loader)
                print(f"[Validation] Epoch {epoch}: AUC={auc_score:.4f}, AUPR={pr_auc_score:.4f}")

                # Early stopping (可选)
                if auc_score > best_auc:
                    best_auc = auc_score
                    patience_counter = 0
                    # torch.save(model.state_dict(), f'best_model_fold{fold}.pth')
                else:
                    patience_counter += 1

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
        print(f"│ AUC:       {auc_score:.4f}")
        print(f"│ AUPR:      {pr_auc_score:.4f}")
        print(f"│ Accuracy:  {acc:.4f}")
        print(f"│ F1-Score:  {f1:.4f}")
        print(f"│ Time:      {fold_time:.1f}s")
        print(f"└─────────────────────────────────────────────")

        # 保存模型 (可选)
        # torch.save(model.state_dict(), f'model_ccl_asps_fold{fold}_final.pth')

    # --- 3. 最终结果统计 ---
    print("\n" + "=" * 70)
    print("FINAL 5-FOLD CV RESULTS (FIXED VERSION)")
    print("=" * 70)
    print(f"AUC:       {np.mean(metrics_history['auc']):.4f} ± {np.std(metrics_history['auc']):.4f}")
    print(f"AUPR:      {np.mean(metrics_history['pr_auc']):.4f} ± {np.std(metrics_history['pr_auc']):.4f}")
    print(f"Accuracy:  {np.mean(metrics_history['acc']):.4f} ± {np.std(metrics_history['acc']):.4f}")
    print(f"Precision: {np.mean(metrics_history['prec']):.4f} ± {np.std(metrics_history['prec']):.4f}")
    print(f"Recall:    {np.mean(metrics_history['rec']):.4f} ± {np.std(metrics_history['rec']):.4f}")
    print(f"F1-Score:  {np.mean(metrics_history['f1']):.4f} ± {np.std(metrics_history['f1']):.4f}")
    print("=" * 70)

    # 修复: 保存完整结果到文件
    results_dict = {
        'auc': metrics_history['auc'],
        'pr_auc': metrics_history['pr_auc'],
        'acc': metrics_history['acc'],
        'f1': metrics_history['f1'],
        'mean_auc': np.mean(metrics_history['auc']),
        'std_auc': np.std(metrics_history['auc']),
        'mean_aupr': np.mean(metrics_history['pr_auc']),
        'std_aupr': np.std(metrics_history['pr_auc']),
    }

    # np.save('cv_results_fixed.npy', results_dict)
    print("\n[Info] Training completed successfully!")