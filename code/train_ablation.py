import sys
import torch
import torch.nn as nn
import numpy as np
from model_ablation import AttnFusionGCNNet_Ablation
from utils import *
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, precision_recall_curve, auc, matthews_corrcoef, confusion_matrix
from torch_geometric.loader import DataLoader
import os
import time
import json
from datetime import datetime

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
    'full',              # 完整模型（基线）
    'no_mirna_seq',      # 消融miRNA序列特征 (m1)
    'no_mirna_cgr',      # 消融miRNA CGR特征 (m2)
    'no_drug_seq',       # 消融drug序列特征 (d1)
    'no_drug_fp',        # 消融drug指纹特征 (d2)
    'no_attention',      # 消融交叉注意力
    'no_contrastive'     # 消融协同对比学习
]

ABLATION_NAMES = {
    'full': 'Full Model (Baseline)',
    'no_mirna_seq': 'w/o miRNA Sequence (m1)',
    'no_mirna_cgr': 'w/o miRNA CGR (m2)',
    'no_drug_seq': 'w/o Drug Sequence (d1)',
    'no_drug_fp': 'w/o Drug Fingerprint (d2)',
    'no_attention': 'w/o Cross Attention',
    'no_contrastive': 'w/o Contrastive Learning'
}

print(f"{'='*80}")
print(f"ABLATION STUDY - All Modes")
print(f"{'='*80}")
print(f"[Config] Total Epochs: {NUM_EPOCHS}")
print(f"[Config] Warmup Epochs: {WARMUP_EPOCHS}")
print(f"[Config] Number of Folds: {NUM_FOLDS}")
print(f"[Config] Ablation Modes: {len(ABLATION_MODES)}")
for mode in ABLATION_MODES:
    print(f"  - {mode}: {ABLATION_NAMES[mode]}")
print(f"{'='*80}\n")

# ============================================================================
# 训练函数
# ============================================================================
def get_contrastive_weight(epoch, warmup_epochs=5):
    """
    Loss 权重的 Warmup 逻辑
    """
    if epoch <= warmup_epochs:
        progress = epoch / warmup_epochs
        return 0.5 * (1 - np.cos(np.pi * progress))
    return 1.0

def train(model, device, train_loader, optimizer, epoch, ablation_mode):
    model.train()
    metrics = {
        'total_loss': 0,
        'bce_loss': 0,
        'mirna_cl_loss': 0,
        'drug_cl_loss': 0
    }
    batch_count = 0
    contrastive_weight_factor = get_contrastive_weight(epoch, WARMUP_EPOCHS)
    
    # 如果消融对比学习，则不计算对比损失
    use_contrastive = (ablation_mode != 'no_contrastive')

    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)

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

        labels = data.y.view(-1, 1).float().to(device)
        output = output.view(-1, 1)
        output = torch.clamp(output, min=1e-7, max=1.0 - 1e-7)
        
        loss_bce = loss_fn(output, labels)
        
        if use_contrastive:
            loss_mirna_contrastive = loss_dict['contrastive_mirna']
            loss_drug_contrastive = loss_dict['contrastive_drug']
            loss = (GAMMA * loss_bce +
                    contrastive_weight_factor * (ALPHA * loss_mirna_contrastive +
                                                 BETA * loss_drug_contrastive))
        else:
            loss = loss_bce
            loss_mirna_contrastive = torch.tensor(0.0)
            loss_drug_contrastive = torch.tensor(0.0)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n[Error] Loss is NaN/Inf at Epoch {epoch}, Batch {batch_idx}")
            return None

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        metrics['total_loss'] += loss.item()
        metrics['bce_loss'] += loss_bce.item()
        metrics['mirna_cl_loss'] += loss_mirna_contrastive.item()
        metrics['drug_cl_loss'] += loss_drug_contrastive.item()
        batch_count += 1

    return {k: v / batch_count for k, v in metrics.items()}

def predicting(model, device, loader, ablation_mode):
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

    # --- 计算指标 ---
    
    # 1. Accuracy
    acc = accuracy_score(total_labels, total_preds)
    
    # 2. MCC (Matthews Correlation Coefficient)
    mcc = matthews_corrcoef(total_labels, total_preds)
    
    # 3. Sensitivity (Recall) & Specificity
    cm = confusion_matrix(total_labels, total_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    sen = recall_score(total_labels, total_preds, zero_division=0) # Recall
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # Specificity

    # 4. AUC
    try:
        roc_auc = roc_auc_score(total_labels, total_probs)
    except ValueError:
        roc_auc = 0.5

    # 5. AUPR
    precision_vals, recall_vals, _ = precision_recall_curve(total_labels, total_probs)
    pr_auc = auc(recall_vals, precision_vals)

    return acc, mcc, sen, spe, roc_auc, pr_auc

# ============================================================================
# 单个消融模式训练
# ============================================================================
def run_ablation_experiment(ablation_mode, device):
    """
    运行单个消融实验模式的完整5折交叉验证
    """
    print(f"\n{'='*80}")
    print(f"Running Ablation: {ABLATION_NAMES[ablation_mode]}")
    print(f"Mode: {ablation_mode}")
    print(f"{'='*80}")
    
    metrics_history = {
        'acc': [], 'mcc': [], 'sen': [], 'spe': [], 'auc': [], 'pr_auc': []
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

        print(f"{'Epoch':<5} | {'TotLoss':<7} {'BCE':<7} {'miR_CL':<7} {'Drug_CL':<7} | {'AUC':<7} {'AUPR':<7} {'Acc':<7} {'MCC':<7} {'Sen':<7} {'Spe':<7}")
        print("-" * 115)

        for epoch in range(1, NUM_EPOCHS + 1):
            train_metrics = train(model, device, train_loader, optimizer, epoch, ablation_mode)
            
            if train_metrics is None: 
                break

            scheduler.step()
            
            # 获取测试指标
            acc, mcc, sen, spe, auc_score, pr_auc_score = predicting(model, device, test_loader, ablation_mode)
            
            print(f"{epoch:<5} | "
                  f"{train_metrics['total_loss']:.4f}  {train_metrics['bce_loss']:.4f}  "
                  f"{train_metrics['mirna_cl_loss']:.4f}  {train_metrics['drug_cl_loss']:.4f}   | "
                  f"{auc_score:.4f}  {pr_auc_score:.4f}  {acc:.4f}  {mcc:.4f}  {sen:.4f}  {spe:.4f}")

        # --- Fold 结束 ---
        print(f"\n[Fold {fold + 1} Final] (Using Last Epoch Model)")
        acc, mcc, sen, spe, auc_score, pr_auc_score = predicting(model, device, test_loader, ablation_mode)
        
        print(f"Result -> Acc: {acc:.4f}, MCC: {mcc:.4f}, Sen: {sen:.4f}, Spe: {spe:.4f}, AUC: {auc_score:.4f}, AUPR: {pr_auc_score:.4f}")

        metrics_history['acc'].append(acc)
        metrics_history['mcc'].append(mcc)
        metrics_history['sen'].append(sen)
        metrics_history['spe'].append(spe)
        metrics_history['auc'].append(auc_score)
        metrics_history['pr_auc'].append(pr_auc_score)

    # 计算该消融模式的统计结果
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

    print(f"\n{'='*80}")
    print(f"RESULTS for {ABLATION_NAMES[ablation_mode]}")
    print(f"{'='*80}")
    print(f"Acc:  {results['acc_mean']:.4f} ± {results['acc_std']:.4f}")
    print(f"MCC:  {results['mcc_mean']:.4f} ± {results['mcc_std']:.4f}")
    print(f"Sen:  {results['sen_mean']:.4f} ± {results['sen_std']:.4f}")
    print(f"Spe:  {results['spe_mean']:.4f} ± {results['spe_std']:.4f}")
    print(f"AUC:  {results['auc_mean']:.4f} ± {results['auc_std']:.4f}")
    print(f"AUPR: {results['pr_auc_mean']:.4f} ± {results['pr_auc_std']:.4f}")
    print(f"{'='*80}\n")

    return results

# ============================================================================
# 主程序 - 运行所有消融实验
# ============================================================================
if __name__ == "__main__":
    cuda_name = "cuda:0"
    if len(sys.argv) > 1:
        cuda_name = "cuda:" + str(int(sys.argv[1]))

    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}\n')
    
    # 存储所有消融实验结果
    all_results = []
    
    # 运行所有消融实验
    start_time = time.time()
    
    for ablation_mode in ABLATION_MODES:
        mode_start_time = time.time()
        
        try:
            results = run_ablation_experiment(ablation_mode, device)
            all_results.append(results)
        except Exception as e:
            print(f"\n[ERROR] Failed to run ablation mode '{ablation_mode}': {str(e)}")
            import traceback
            traceback.print_exc()
            continue
        
        mode_elapsed = time.time() - mode_start_time
        print(f"[Time] Mode '{ablation_mode}' completed in {mode_elapsed/60:.2f} minutes\n")
    
    total_elapsed = time.time() - start_time
    
    # ============================================================================
    # 生成汇总报告
    # ============================================================================
    print("\n" + "="*100)
    print("ABLATION STUDY - FINAL SUMMARY")
    print("="*100)
    print(f"{'Mode':<25} | {'Acc':<15} {'MCC':<15} {'Sen':<15} {'Spe':<15} {'AUC':<15} {'AUPR':<15}")
    print("-"*100)
    
    for result in all_results:
        print(f"{result['name']:<25} | "
              f"{result['acc_mean']:.4f}±{result['acc_std']:.4f}  "
              f"{result['mcc_mean']:.4f}±{result['mcc_std']:.4f}  "
              f"{result['sen_mean']:.4f}±{result['sen_std']:.4f}  "
              f"{result['spe_mean']:.4f}±{result['spe_std']:.4f}  "
              f"{result['auc_mean']:.4f}±{result['auc_std']:.4f}  "
              f"{result['pr_auc_mean']:.4f}±{result['pr_auc_std']:.4f}")
    
    print("="*100)
    print(f"[Total Time] All ablation experiments completed in {total_elapsed/60:.2f} minutes")
    print("="*100)
    
    # ============================================================================
    # 保存结果到文件
    # ============================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 保存JSON格式（完整数据）
    json_filename = f"ablation_results_{timestamp}.json"
    with open(json_filename, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'config': {
                'num_epochs': NUM_EPOCHS,
                'num_folds': NUM_FOLDS,
                'learning_rate': LR,
                'batch_size': TRAIN_BATCH_SIZE,
                'temperature': TEMPERATURE,
                'alpha': ALPHA,
                'beta': BETA,
                'gamma': GAMMA
            },
            'results': all_results
        }, f, indent=2)
    
    print(f"\n[Saved] Detailed results saved to: {json_filename}")
    
    # 2. 保存TXT格式（易读表格）
    txt_filename = f"ablation_results_{timestamp}.txt"
    with open(txt_filename, 'w') as f:
        f.write("="*100 + "\n")
        f.write("ABLATION STUDY - FINAL SUMMARY\n")
        f.write("="*100 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Total Time: {total_elapsed/60:.2f} minutes\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"{'Mode':<25} | {'Acc':<15} {'MCC':<15} {'Sen':<15} {'Spe':<15} {'AUC':<15} {'AUPR':<15}\n")
        f.write("-"*100 + "\n")
        
        for result in all_results:
            f.write(f"{result['name']:<25} | "
                   f"{result['acc_mean']:.4f}±{result['acc_std']:.4f}  "
                   f"{result['mcc_mean']:.4f}±{result['mcc_std']:.4f}  "
                   f"{result['sen_mean']:.4f}±{result['sen_std']:.4f}  "
                   f"{result['spe_mean']:.4f}±{result['spe_std']:.4f}  "
                   f"{result['auc_mean']:.4f}±{result['auc_std']:.4f}  "
                   f"{result['pr_auc_mean']:.4f}±{result['pr_auc_std']:.4f}\n")
        
        f.write("="*100 + "\n\n")
        
        # 详细的各Fold结果
        f.write("DETAILED FOLD RESULTS\n")
        f.write("="*100 + "\n")
        for result in all_results:
            f.write(f"\n{result['name']}:\n")
            f.write("-"*50 + "\n")
            for i in range(NUM_FOLDS):
                f.write(f"Fold {i+1}: ")
                f.write(f"Acc={result['fold_results']['acc'][i]:.4f}, ")
                f.write(f"MCC={result['fold_results']['mcc'][i]:.4f}, ")
                f.write(f"Sen={result['fold_results']['sen'][i]:.4f}, ")
                f.write(f"Spe={result['fold_results']['spe'][i]:.4f}, ")
                f.write(f"AUC={result['fold_results']['auc'][i]:.4f}, ")
                f.write(f"AUPR={result['fold_results']['pr_auc'][i]:.4f}\n")
    
    print(f"[Saved] Summary table saved to: {txt_filename}")
    
    # 3. 保存CSV格式（方便导入Excel）
    csv_filename = f"ablation_results_{timestamp}.csv"
    with open(csv_filename, 'w') as f:
        f.write("Mode,Metric,Mean,Std\n")
        for result in all_results:
            mode_name = result['name']
            f.write(f"{mode_name},Accuracy,{result['acc_mean']:.4f},{result['acc_std']:.4f}\n")
            f.write(f"{mode_name},MCC,{result['mcc_mean']:.4f},{result['mcc_std']:.4f}\n")
            f.write(f"{mode_name},Sensitivity,{result['sen_mean']:.4f},{result['sen_std']:.4f}\n")
            f.write(f"{mode_name},Specificity,{result['spe_mean']:.4f},{result['spe_std']:.4f}\n")
            f.write(f"{mode_name},AUC,{result['auc_mean']:.4f},{result['auc_std']:.4f}\n")
            f.write(f"{mode_name},AUPR,{result['pr_auc_mean']:.4f},{result['pr_auc_std']:.4f}\n")
    
    print(f"[Saved] CSV format saved to: {csv_filename}")
    
    print("\n[Complete] All ablation experiments finished successfully!")
    print(f"[Output Files]")
    print(f"  - {json_filename} (detailed JSON)")
    print(f"  - {txt_filename} (formatted text)")
    print(f"  - {csv_filename} (Excel-friendly CSV)")