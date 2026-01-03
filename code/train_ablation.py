import sys
import torch
import torch.nn as nn
import numpy as np
from model_ablation import AttnFusionGCNNet_Ablation
from utils import *
# 确保导入 auc 和 roc_curve
from sklearn.metrics import accuracy_score, recall_score, precision_recall_curve, auc, matthews_corrcoef, \
    confusion_matrix, roc_curve
from torch_geometric.loader import DataLoader
import os
import time
import json
import argparse
from datetime import datetime
import matplotlib.pyplot as plt

# 防止在无图形界面的服务器上报错
plt.switch_backend('agg')

# --- 全局常量 ---
LOG_INTERVAL = 45
NUM_FOLDS = 5
loss_fn = nn.BCELoss()

# ============================================================================
# 超参数配置
# ============================================================================
LR = 0.0005
WEIGHT_DECAY = 0.0032
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
NUM_EPOCHS = 30
WARMUP_EPOCHS = 5

# --- 对比学习参数 ---
ALPHA = 0.5
BETA = 0.5
GAMMA = 1.0
TEMPERATURE = 0.1
LAM = 0.5
CONTRASTIVE_DIM = 128

# ============================================================================
# 消融实验模式配置
# ============================================================================
ABLATION_MODES = [
    'full',  # 完整模型（基线）
    'no_mirna_seq',  # 消融miRNA序列特征 (m1)
    'no_mirna_cgr',  # 消融miRNA CGR特征 (m2)
    'no_drug_seq',  # 消融drug序列特征 (d1)
    'no_drug_fp',  # 消融drug指纹特征 (d2)
    'no_attention',  # 消融交叉注意力
    'no_contrastive',  # 消融协同对比学习
    'no_mirna_seq_drug_seq',  # m1+d1
    'no_mirna_cgr_drug_fp'  # m2+d2
]

ABLATION_NAMES = {
    'full': 'Full Model (Baseline)',
    'no_mirna_seq': 'w/o miRNA Sequence (m1)',
    'no_mirna_cgr': 'w/o miRNA CGR (m2)',
    'no_drug_seq': 'w/o Drug Sequence (d1)',
    'no_drug_fp': 'w/o Drug Fingerprint (d2)',
    'no_attention': 'w/o Cross Attention',
    'no_contrastive': 'w/o Contrastive Learning',
    'no_mirna_seq_drug_seq': 'w/o miRNA Seq + Drug Seq (m1+d1)',
    'no_mirna_cgr_drug_fp': 'w/o miRNA CGR + Drug FP (m2+d2)'
}


# ============================================================================
# 训练函数
# ============================================================================
def get_contrastive_weight(epoch, warmup_epochs=5):
    if epoch <= warmup_epochs:
        progress = epoch / warmup_epochs
        return 0.5 * (1 - np.cos(np.pi * progress))
    return 1.0


def train(model, device, train_loader, optimizer, epoch, ablation_mode):
    model.train()
    metrics = {'total_loss': 0, 'bce_loss': 0, 'mirna_cl_loss': 0, 'drug_cl_loss': 0}
    batch_count = 0
    contrastive_weight_factor = get_contrastive_weight(epoch, WARMUP_EPOCHS)
    use_contrastive = (ablation_mode != 'no_contrastive')

    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)

        if use_contrastive:
            output, loss_dict = model(data, current_epoch=epoch, total_epochs=NUM_EPOCHS,
                                      warmup_epochs=WARMUP_EPOCHS, return_contrastive_loss=True)
        else:
            output = model(data, current_epoch=epoch, total_epochs=NUM_EPOCHS,
                           warmup_epochs=WARMUP_EPOCHS, return_contrastive_loss=False)

        labels = data.y.view(-1, 1).float().to(device)
        output = output.view(-1, 1)
        output = torch.clamp(output, min=1e-7, max=1.0 - 1e-7)

        loss_bce = loss_fn(output, labels)

        if use_contrastive:
            loss = (GAMMA * loss_bce + contrastive_weight_factor * (
                    ALPHA * loss_dict['contrastive_mirna'] + BETA * loss_dict['contrastive_drug']))
            loss_mirna = loss_dict['contrastive_mirna']
            loss_drug = loss_dict['contrastive_drug']
        else:
            loss = loss_bce
            loss_mirna = torch.tensor(0.0)
            loss_drug = torch.tensor(0.0)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n[Error] Loss is NaN/Inf at Epoch {epoch}, Batch {batch_idx}")
            return None

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        metrics['total_loss'] += loss.item()
        metrics['bce_loss'] += loss_bce.item()
        metrics['mirna_cl_loss'] += loss_mirna.item()
        metrics['drug_cl_loss'] += loss_drug.item()
        batch_count += 1

    return {k: v / batch_count for k, v in metrics.items()}


def predicting(model, device, loader, ablation_mode, return_curves=False):
    model.eval()
    total_probs = []
    total_labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data, current_epoch=0, total_epochs=NUM_EPOCHS,
                           warmup_epochs=WARMUP_EPOCHS, return_contrastive_loss=False)
            total_probs.extend(output.cpu().numpy().flatten())
            total_labels.extend(data.y.view(-1, 1).cpu().numpy().flatten())

    total_probs = np.array(total_probs)
    total_labels = np.array(total_labels)
    total_preds = (total_probs >= 0.5).astype(int)

    acc = accuracy_score(total_labels, total_preds)
    mcc = matthews_corrcoef(total_labels, total_preds)
    cm = confusion_matrix(total_labels, total_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sen = recall_score(total_labels, total_preds, zero_division=0)
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # 这里依然计算单次的AUC，用于在控制台打印进度
    fpr_vals, tpr_vals, _ = roc_curve(total_labels, total_probs)
    roc_auc = auc(fpr_vals, tpr_vals)

    precision_vals, recall_vals, _ = precision_recall_curve(total_labels, total_probs)
    pr_auc = auc(recall_vals, precision_vals)

    if return_curves:
        curve_data = {
            'fpr': fpr_vals.tolist(),
            'tpr': tpr_vals.tolist(),
            'precision': precision_vals.tolist(),
            'recall': recall_vals.tolist()
        }
        return acc, mcc, sen, spe, roc_auc, pr_auc, curve_data

    return acc, mcc, sen, spe, roc_auc, pr_auc


# ============================================================================
# 绘图函数 (核心修改)
# ============================================================================
def plot_and_save_curves(all_results, output_dir, file_prefix):
    """
    绘制并保存 ROC 和 PR 曲线
    只绘制 Mean Curve，并基于 Mean Curve 计算面积。
    """
    if not all_results:
        return

    result = all_results[0]
    mode_name = result['name']
    fold_curves = result['fold_results']['curves']

    # ----------------------------------------------------------------
    # 1. 绘制 ROC 曲线 (Mean FPR vs Mean TPR)
    # ----------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    tprs = []
    # 定义公共的 mean_fpr，用于插值对齐
    mean_fpr = np.linspace(0, 1, 100)

    for fold_data in fold_curves:
        fpr = np.array(fold_data['data']['fpr'])
        tpr = np.array(fold_data['data']['tpr'])

        # 插值：计算对应 mean_fpr 位置上的 tpr
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    # 计算 Mean TPR
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    # 【关键】根据 Mean Curve 计算 AUC
    mean_auc = auc(mean_fpr, mean_tpr)

    # 绘制
    plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.4f})', lw=2.5)

    # 随机猜测线
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {mode_name}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    roc_filename = os.path.join(output_dir, f"{file_prefix}_ROC.png")
    plt.savefig(roc_filename, dpi=300)
    plt.close()

    # ----------------------------------------------------------------
    # 2. 绘制 PR 曲线 (Mean Recall vs Mean Precision)
    # ----------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    prs = []
    # 定义公共的 mean_recall
    mean_recall = np.linspace(0, 1, 100)

    for fold_data in fold_curves:
        precision = np.array(fold_data['data']['precision'])
        recall = np.array(fold_data['data']['recall'])

        # PR曲线插值需要 recall 单调递增
        # 通常 recall 是从 1 到 0 (或 0 到 1)，需要确保反转正确
        reversed_recall = recall[::-1]
        reversed_precision = precision[::-1]

        # 插值：计算对应 mean_recall 位置上的 precision
        interp_precision = np.interp(mean_recall, reversed_recall, reversed_precision)
        prs.append(interp_precision)

    # 计算 Mean Precision
    mean_precision = np.mean(prs, axis=0)

    # 【关键】根据 Mean Curve 计算 AUPR
    mean_aupr = auc(mean_recall, mean_precision)

    # 绘制
    plt.plot(mean_recall, mean_precision, color='b', label=f'Mean PR (AUPR = {mean_aupr:.4f})', lw=2.5)

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR Curve - {mode_name}')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)

    pr_filename = os.path.join(output_dir, f"{file_prefix}_PR.png")
    plt.savefig(pr_filename, dpi=300)
    plt.close()

    print(f"[Plot] Saved Mean Curve plots to:")
    print(f"  - {roc_filename}")
    print(f"  - {pr_filename}")
    print(f"[Result] Calculated Area Under Mean Curve:")
    print(f"  - Mean ROC AUC : {mean_auc:.4f}")
    print(f"  - Mean PR AUPR : {mean_aupr:.4f}")


# ============================================================================
# 单个消融模式训练
# ============================================================================
def run_ablation_experiment(ablation_mode, device):
    print(f"\n{'=' * 80}")
    print(f"Running Ablation: {ABLATION_NAMES[ablation_mode]}")
    print(f"Mode: {ablation_mode}")
    print(f"{'=' * 80}")

    metrics_history = {
        'acc': [], 'mcc': [], 'sen': [], 'spe': [], 'auc': [], 'pr_auc': [],
        'curves': []
    }

    for fold in range(NUM_FOLDS):
        print(f"\n>>> Fold {fold + 1}/{NUM_FOLDS}")
        print("-" * 80)

        train_data = TestbedDataset(root='data', dataset='train' + str(fold))
        test_data = TestbedDataset(root='data', dataset='test' + str(fold))
        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, drop_last=False)

        model = AttnFusionGCNNet_Ablation(
            ablation_mode=ablation_mode,
            n_output=1, n_filters=32, embed_dim=64, num_features_xd=78,
            num_features_smile=66, num_features_xt=25, output_dim=128, dropout=0.2,
            contrastive_dim=CONTRASTIVE_DIM, temperature=TEMPERATURE, lam=LAM
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=LR * 0.01)

        print(
            f"{'Epoch':<5} | {'TotLoss':<7} {'BCE':<7} {'miR_CL':<7} {'Drug_CL':<7} | {'AUC':<7} {'AUPR':<7} {'Acc':<7} {'MCC':<7}")
        print("-" * 95)

        for epoch in range(1, NUM_EPOCHS + 1):
            train_metrics = train(model, device, train_loader, optimizer, epoch, ablation_mode)
            if train_metrics is None: break
            scheduler.step()

            acc, mcc, sen, spe, auc_score, pr_auc_score = predicting(model, device, test_loader, ablation_mode,
                                                                     return_curves=False)
            print(f"{epoch:<5} | {train_metrics['total_loss']:.4f}  {train_metrics['bce_loss']:.4f}  "
                  f"{train_metrics['mirna_cl_loss']:.4f}  {train_metrics['drug_cl_loss']:.4f}   | "
                  f"{auc_score:.4f}  {pr_auc_score:.4f}  {acc:.4f}  {mcc:.4f}")

        # Fold 结束，收集曲线数据
        acc, mcc, sen, spe, auc_score, pr_auc_score, curve_data = predicting(model, device, test_loader, ablation_mode,
                                                                             return_curves=True)

        metrics_history['acc'].append(acc)
        metrics_history['mcc'].append(mcc)
        metrics_history['sen'].append(sen)
        metrics_history['spe'].append(spe)
        metrics_history['auc'].append(auc_score)
        metrics_history['pr_auc'].append(pr_auc_score)

        metrics_history['curves'].append({
            'fold': fold + 1,
            'auc': auc_score,
            'pr_auc': pr_auc_score,
            'data': curve_data
        })

    results = {
        'mode': ablation_mode,
        'name': ABLATION_NAMES[ablation_mode],
        'acc_mean': float(np.mean(metrics_history['acc'])),
        'acc_std': float(np.std(metrics_history['acc'])),
        'mcc_mean': float(np.mean(metrics_history['mcc'])),
        'mcc_std': float(np.std(metrics_history['mcc'])),
        'sen_mean': float(np.mean(metrics_history['sen'])),
        'sen_std': float(np.std(metrics_history['sen'])),
        'spe_mean': float(np.mean(metrics_history['spe'])),
        'spe_std': float(np.std(metrics_history['spe'])),
        'auc_mean': float(np.mean(metrics_history['auc'])),
        'auc_std': float(np.std(metrics_history['auc'])),
        'pr_auc_mean': float(np.mean(metrics_history['pr_auc'])),
        'pr_auc_std': float(np.std(metrics_history['pr_auc'])),
        'fold_results': metrics_history
    }
    return results


# ============================================================================
# 主程序
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run specific ablation study mode")
    parser.add_argument('--mode', type=str, required=True,
                        help=f"Ablation mode to run. Options: {ABLATION_MODES}")
    parser.add_argument('--gpu', type=int, default=0,
                        help="CUDA device ID (default: 0)")

    args = parser.parse_args()

    if args.mode not in ABLATION_MODES:
        print(f"[Error] Invalid mode: {args.mode}")
        sys.exit(1)

    cuda_name = f"cuda:{args.gpu}"
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    print(f"Selected Ablation Mode: {args.mode}")

    start_time = time.time()
    all_results = []

    try:
        results = run_ablation_experiment(args.mode, device)
        all_results.append(results)
    except Exception as e:
        print(f"\n[ERROR] Failed to run ablation mode '{args.mode}': {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    total_elapsed = time.time() - start_time

    # ============================================================================
    # 保存结果与绘图
    # ============================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = "ablation"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_prefix = f"ablation_{args.mode}_{timestamp}"

    print("\n" + "=" * 80)
    print(f"SUMMARY FOR {args.mode}")
    print("=" * 80)

    if all_results:
        r = all_results[0]
        print(f"Acc:  {r['acc_mean']:.4f} ± {r['acc_std']:.4f}")
        print(f"MCC:  {r['mcc_mean']:.4f} ± {r['mcc_std']:.4f}")
        print(f"AUC:  {r['auc_mean']:.4f} ± {r['auc_std']:.4f}")
        print(f"AUPR: {r['pr_auc_mean']:.4f} ± {r['pr_auc_std']:.4f}")

        # 1. 保存 JSON (含详细曲线数据)
        json_filename = os.path.join(output_dir, f"{file_prefix}.json")
        with open(json_filename, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'mode': args.mode,
                'config': {'epochs': NUM_EPOCHS, 'lr': LR},
                'results': all_results
            }, f, indent=2)

        # 2. 保存 CSV
        csv_filename = os.path.join(output_dir, f"{file_prefix}.csv")
        with open(csv_filename, 'w') as f:
            f.write("Mode,Metric,Mean,Std\n")
            f.write(f"{args.mode},Accuracy,{r['acc_mean']:.4f},{r['acc_std']:.4f}\n")
            f.write(f"{args.mode},MCC,{r['mcc_mean']:.4f},{r['mcc_std']:.4f}\n")
            f.write(f"{args.mode},AUC,{r['auc_mean']:.4f},{r['auc_std']:.4f}\n")
            f.write(f"{args.mode},AUPR,{r['pr_auc_mean']:.4f},{r['pr_auc_std']:.4f}\n")

        print(f"\n[Saved] Results saved to folder '{output_dir}':")
        print(f"  - {json_filename}")
        print(f"  - {csv_filename}")

        # 3. 绘制并保存曲线图片 (现在只画一条 Mean Curve 并计算面积)
        print("\n[Plotting] Generating Mean ROC and Mean PR curves...")
        plot_and_save_curves(all_results, output_dir, file_prefix)

    else:
        print("\n[Warning] No results to save.")