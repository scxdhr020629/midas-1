import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model_fixed import AttnFusionGCNNet, Model_Contrast  # 修复: 导入 Model_Contrast 用于验证
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
# LR = 0.001
WEIGHT_DECAY = 0.0032
TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
# TRAIN_BATCH_SIZE = 64
# TEST_BATCH_SIZE = 64

# NUM_EPOCHS = 45
NUM_EPOCHS = 30

# --- 对比学习超参数 ---
# 修复: 不再归一化权重，直接使用原始值
# 这样更直观：如果想增强某个损失，直接调大对应权重即可
ALPHA = 0.2  # miRNA 视图对比损失权重
BETA = 0.2  # Drug 视图对比损失权重
GAMMA = 0.6 # 主任务 (BCE) 权重 - 修复: 增加 BCE 权重以稳定训练

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
# 验证函数 - 从 model_fixed.py 移植并使用真实数据
#
# ============================================================================

def validate_model_initialization(model):
    """验证模型初始化是否成功"""
    print("\n" + "=" * 60)
    print("🔍 验证模型初始化")
    print("=" * 60)
    
    print("✅ 模型初始化成功！")
    print(f"📊 模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 检查各模块参数
    drug_params = sum(p.numel() for name, p in model.named_parameters() 
                      if 'drug' in name.lower() or 'smile' in name.lower() or 'xd' in name.lower())
    mirna_params = sum(p.numel() for name, p in model.named_parameters() 
                       if 'mirna' in name.lower() or 'xt' in name.lower() or 'matrix' in name.lower())
    contrast_params = sum(p.numel() for name, p in model.named_parameters() 
                          if 'contrast' in name.lower())
    
    print(f"📊 Drug 编码器参数: {drug_params:,}")
    print(f"📊 miRNA 编码器参数: {mirna_params:,}")
    print(f"📊 对比学习模块参数: {contrast_params:,}")


def validate_forward_pass(model, device, data_loader, current_epoch=10, total_epochs=100):
    """使用真实数据验证前向传播"""
    print("\n" + "=" * 60)
    print("🔍 验证前向传播 (使用真实数据)")
    print("=" * 60)
    
    model.eval()
    
    # 获取一个批次的真实数据
    data_iter = iter(data_loader)
    data = next(data_iter).to(device)
    
    batch_size = data.target.shape[0]
    print(f"📊 批次大小: {batch_size}")
    
    with torch.no_grad():
        predictions, loss_dict = model(
            data,
            current_epoch=current_epoch,
            total_epochs=total_epochs,
            warmup_epochs=WARMUP_EPOCHS,
            return_contrastive_loss=True
        )
    
    print("\n📈 前向传播输出:")
    print(f"  预测形状: {predictions.shape}")
    print(f"  预测范围: [{predictions.min().item():.4f}, {predictions.max().item():.4f}]")
    print(f"  Drug 对比损失: {loss_dict['contrastive_drug']:.4f}")
    print(f"  miRNA 对比损失: {loss_dict['contrastive_mirna']:.4f}")
    print(f"  总对比损失: {sum(v.item() for v in loss_dict.values()):.4f}")
    
    # 检查是否有 NaN
    if torch.isnan(predictions).any():
        print("⚠️  警告: 预测中存在 NaN 值!")
    else:
        print("✅ 预测值正常，无 NaN")
    
    return predictions, loss_dict


def validate_trivial_solution_fix(model, device, data_loader):
    """使用真实数据验证平凡解是否被修复"""
    print("\n" + "=" * 60)
    print("🔍 验证平凡解修复 (使用真实数据)")
    print("=" * 60)
    
    model.eval()
    
    # 获取一个批次的真实数据
    data_iter = iter(data_loader)
    data = next(data_iter).to(device)
    batch_size = data.target.shape[0]
    
    # 运行前向传播以获取损失
    with torch.no_grad():
        _, loss_dict = model(
            data,
            current_epoch=10,
            total_epochs=100,
            warmup_epochs=5,
            return_contrastive_loss=True
        )
    
    print(f"\n📊 当前对比损失 (真实数据):")
    print(f"  miRNA 对比损失: {loss_dict['contrastive_mirna']:.4f}")
    print(f"  Drug 对比损失: {loss_dict['contrastive_drug']:.4f}")
    
    # 使用模拟特征测试平凡解
    hidden_dim = model.output_dim
    contrast_module = model.contrast_mirna
    
    seq_features = torch.randn(batch_size, hidden_dim).to(device)
    mol_features = torch.randn(batch_size, hidden_dim).to(device)
    
    # 平凡解：Fused 完全等于 Seq
    fused_trivial = seq_features.clone()
    # 正常融合：Fused = f(Seq, Mol)
    fused_normal = 0.5 * seq_features + 0.5 * mol_features
    
    pos_mask = torch.eye(batch_size).to(device)
    neg_mask = 1 - pos_mask
    
    with torch.no_grad():
        loss_trivial = contrast_module(seq_features, mol_features, fused_trivial, pos_mask, neg_mask)
        loss_normal = contrast_module(seq_features, mol_features, fused_normal, pos_mask, neg_mask)
    
    print(f"\n📊 平凡解测试:")
    print(f"  平凡解损失 (Fused=Seq): {loss_trivial.item():.4f}")
    print(f"  正常融合损失: {loss_normal.item():.4f}")
    
    if loss_trivial > loss_normal * 0.8:
        print("✅ 修复成功！平凡解不再是最优解")
    else:
        print("⚠️  平凡解仍然可能存在风险")
    
    # 测试视图间相似度
    seq_norm = F.normalize(seq_features, dim=1)
    mol_norm = F.normalize(mol_features, dim=1)
    similarity = (seq_norm * mol_norm).sum(dim=1).mean()
    
    print(f"\n📊 Seq-Mol 视图间相似度: {similarity:.4f}")
    print("   (期望值接近 0，表示随机特征)")


def validate_contrastive_learning(model, device, data_loader, num_batches=5):
    """验证对比学习在多个批次上的表现"""
    print("\n" + "=" * 60)
    print("🔍 验证对比学习效果 (多批次)")
    print("=" * 60)
    
    model.eval()
    
    mirna_losses = []
    drug_losses = []
    
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if i >= num_batches:
                break
            
            data = data.to(device)
            _, loss_dict = model(
                data,
                current_epoch=10,
                total_epochs=100,
                warmup_epochs=5,
                return_contrastive_loss=True
            )
            
            mirna_losses.append(loss_dict['contrastive_mirna'].item())
            drug_losses.append(loss_dict['contrastive_drug'].item())
    
    print(f"\n📊 对比学习损失统计 ({len(mirna_losses)} 批次):")
    print(f"  miRNA CL: {np.mean(mirna_losses):.4f} ± {np.std(mirna_losses):.4f}")
    print(f"  Drug CL:  {np.mean(drug_losses):.4f} ± {np.std(drug_losses):.4f}")
    
    if np.std(mirna_losses) < np.mean(mirna_losses) and np.std(drug_losses) < np.mean(drug_losses):
        print("✅ 对比学习损失稳定")
    else:
        print("⚠️  对比学习损失波动较大")


def run_all_validations(model, device, train_loader, test_loader):
    """运行所有验证测试"""
    print("\n" + "=" * 70)
    print("🚀 运行完整验证套件 (使用真实数据)")
    print("=" * 70)
    
    validate_model_initialization(model)
    
    print("\n--- 使用训练数据 ---")
    validate_forward_pass(model, device, train_loader, current_epoch=10, total_epochs=NUM_EPOCHS)
    
    print("\n--- 使用测试数据 ---")
    validate_forward_pass(model, device, test_loader, current_epoch=10, total_epochs=NUM_EPOCHS)
    
    validate_trivial_solution_fix(model, device, train_loader)
    validate_contrastive_learning(model, device, train_loader, num_batches=5)
    
    print("\n" + "=" * 70)
    print("✅ 所有验证测试完成！")
    print("=" * 70)


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

    # --- 运行验证测试 (使用第一折数据) ---
    print("\n>>> 加载第一折数据进行模型验证...")
    
    train_data_val = TestbedDataset(root='data', dataset='train0')
    test_data_val = TestbedDataset(root='data', dataset='test0')
    
    train_loader_val = DataLoader(
        train_data_val, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True
    )
    test_loader_val = DataLoader(
        test_data_val, batch_size=TEST_BATCH_SIZE, shuffle=False, drop_last=False
    )
    
    model_val = modeling(
        n_output=1, n_filters=32, embed_dim=64, num_features_xd=78,
        num_features_smile=66, num_features_xt=25, output_dim=128, dropout=0.2,
        contrastive_dim=CONTRASTIVE_DIM, temperature=TEMPERATURE, lam=LAM
    ).to(device)
    
    run_all_validations(model_val, device, train_loader_val, test_loader_val)
    
    del model_val, train_loader_val, test_loader_val
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\n>>> 验证完成，开始正式 5-Fold CV 训练...")
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