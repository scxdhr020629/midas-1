import sys
import torch
import torch.nn as nn
import numpy as np
from paramter_find_model import AttnFusionGCNNet
from utils import *
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, precision_recall_curve, auc, matthews_corrcoef, \
    confusion_matrix
from torch_geometric.loader import DataLoader
import os
import json
import time
from datetime import datetime

# 创建结果保存文件夹
RESULTS_DIR = "parameter_MDR"
os.makedirs(RESULTS_DIR, exist_ok=True)

# 全局常量
NUM_FOLDS = 5
loss_fn = nn.BCELoss()

# ============================================================================
# 基础超参数配置（默认值）
# ============================================================================
DEFAULT_CONFIG = {
    'LR': 0.0005,
    'WEIGHT_DECAY': 0.0032,
    'TRAIN_BATCH_SIZE': 64,
    'TEST_BATCH_SIZE': 64,
    'NUM_EPOCHS': 30,
    'WARMUP_EPOCHS': 5,
    'ALPHA': 0.5,
    'BETA': 0.5,
    'GAMMA': 1.0,
    'TEMPERATURE': 0.1,
    'LAM': 0.5,
    'CONTRASTIVE_DIM': 128,
    'EMBED_DIM': 64,  # <--- 新增：默认初始嵌入维度
    'CONV_KERNELS': [4, 3, 2]  # 多尺度卷积核
}

# ============================================================================
# 调参配置
# ============================================================================
PARAM_GRIDS = {
    # 'alpha_beta': {
    #     'name': 'Alpha_Beta_Balance',
    #     'values': [0.1, 0.3, 0.5, 0.7, 0.9],
    #     'description': 'Testing α=β values for contrastive learning weight balance'
    # },
    'dimension': {
        'name': 'Feature_Dimension',
        'values': [32, 64, 128, 256, 512],  # <--- 这里定义你要搜索的初始嵌入维度
        'description': 'Testing initial embedding dimension'
    }
    # 'epochs': {
    #     'name': 'Training_Epochs',
    #     'values': [10, 20, 30, 40, 50],
    #     'description': 'Testing number of training epochs'
    # },
    # 'conv_kernels': {
    #     'name': 'Multi_Scale_Kernels',
    #     'values': [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]],
    #     'description': 'Testing multi-scale convolution kernel sizes'
    # }
}


# ============================================================================
# 训练和评估函数
# ============================================================================
def get_contrastive_weight(epoch, warmup_epochs=5):
    if epoch <= warmup_epochs:
        progress = epoch / warmup_epochs
        return 0.5 * (1 - np.cos(np.pi * progress))
    return 1.0


def train(model, device, train_loader, optimizer, epoch, config):
    model.train()
    metrics = {
        'total_loss': 0,
        'bce_loss': 0,
        'mirna_cl_loss': 0,
        'drug_cl_loss': 0
    }
    batch_count = 0
    contrastive_weight_factor = get_contrastive_weight(epoch, config['WARMUP_EPOCHS'])

    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)

        output, loss_dict = model(
            data,
            current_epoch=epoch,
            total_epochs=config['NUM_EPOCHS'],
            warmup_epochs=config['WARMUP_EPOCHS'],
            return_contrastive_loss=True
        )

        labels = data.y.view(-1, 1).float().to(device)
        output = output.view(-1, 1)
        output = torch.clamp(output, min=1e-7, max=1.0 - 1e-7)

        loss_bce = loss_fn(output, labels)
        loss_mirna_contrastive = loss_dict['contrastive_mirna']
        loss_drug_contrastive = loss_dict['contrastive_drug']

        loss = (config['GAMMA'] * loss_bce +
                contrastive_weight_factor * (config['ALPHA'] * loss_mirna_contrastive +
                                             config['BETA'] * loss_drug_contrastive))

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


def predicting(model, device, loader, config):
    model.eval()
    total_probs = []
    total_labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(
                data,
                current_epoch=0,
                total_epochs=config['NUM_EPOCHS'],
                warmup_epochs=config['WARMUP_EPOCHS'],
                return_contrastive_loss=False
            )
            probs = output.cpu().numpy().flatten()
            total_probs.extend(probs)
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

    try:
        roc_auc = roc_auc_score(total_labels, total_probs)
    except ValueError:
        roc_auc = 0.5

    precision_vals, recall_vals, _ = precision_recall_curve(total_labels, total_probs)
    pr_auc = auc(recall_vals, precision_vals)

    return acc, mcc, sen, spe, roc_auc, pr_auc


# ============================================================================
# 单次实验运行
# ============================================================================
def run_single_experiment(config, device):
    """运行单次实验（5折交叉验证）"""
    metrics_history = {
        'acc': [], 'mcc': [], 'sen': [], 'spe': [], 'auc': [], 'pr_auc': []
    }

    for fold in range(NUM_FOLDS):
        print(f"\n  Fold {fold + 1}/{NUM_FOLDS}")

        train_data = TestbedDataset(root='data', dataset='train' + str(fold))
        test_data = TestbedDataset(root='data', dataset='test' + str(fold))

        train_loader = DataLoader(train_data, batch_size=config['TRAIN_BATCH_SIZE'],
                                  shuffle=True, drop_last=True)
        test_loader = DataLoader(test_data, batch_size=config['TEST_BATCH_SIZE'],
                                 shuffle=False, drop_last=False)

        # 动态创建模型
        model = create_model_with_kernels(config).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config['LR'],
                                      weight_decay=config['WEIGHT_DECAY'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['NUM_EPOCHS'], eta_min=config['LR'] * 0.01
        )

        # 训练循环
        for epoch in range(1, config['NUM_EPOCHS'] + 1):
            train_metrics = train(model, device, train_loader, optimizer, epoch, config)
            if train_metrics is None:
                break
            scheduler.step()

        # 最终评估
        acc, mcc, sen, spe, auc_score, pr_auc_score = predicting(model, device, test_loader, config)

        metrics_history['acc'].append(acc)
        metrics_history['mcc'].append(mcc)
        metrics_history['sen'].append(sen)
        metrics_history['spe'].append(spe)
        metrics_history['auc'].append(auc_score)
        metrics_history['pr_auc'].append(pr_auc_score)

    # 计算平均值和标准差
    results = {
        'mean': {k: np.mean(v) for k, v in metrics_history.items()},
        'std': {k: np.std(v) for k, v in metrics_history.items()},
        'raw': metrics_history
    }

    return results


def create_model_with_kernels(config):
    """根据配置创建模型"""
    # 修正：现在使用 config['EMBED_DIM'] 来设置初始嵌入维度
    return AttnFusionGCNNet(
        n_output=1,
        n_filters=32,
        embed_dim=config['EMBED_DIM'],  # <--- 【关键修改】这里读取配置中的嵌入维度
        num_features_xd=78,
        num_features_smile=66,
        num_features_xt=25,
        output_dim=128,  # 这里的 output_dim 是卷积后的维度，这里暂时保持不变
        dropout=0.2,
        contrastive_dim=config['CONTRASTIVE_DIM'],
        temperature=config['TEMPERATURE'],
        lam=config['LAM']
    )


# ============================================================================
# 主调参流程
# ============================================================================
def tune_hyperparameters(device):
    """执行完整的超参数调优流程"""
    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("HYPERPARAMETER TUNING - MDR MODEL")
    print("=" * 80)

    # 1. 调整 Alpha & Beta
    # print("\n" + "="*80)
    # print("EXPERIMENT 1: Alpha & Beta Balance (α=β)")
    # print("="*80)
    # alpha_beta_results = {}

    # for val in PARAM_GRIDS['alpha_beta']['values']:
    #     config = DEFAULT_CONFIG.copy()
    #     config['ALPHA'] = val
    #     config['BETA'] = val

    #     print(f"\n>>> Testing α=β={val}")
    #     results = run_single_experiment(config, device)
    #     alpha_beta_results[val] = results

    #     print(f"  AUC: {results['mean']['auc']:.4f} ± {results['std']['auc']:.4f}")
    #     print(f"  AUPR: {results['mean']['pr_auc']:.4f} ± {results['std']['pr_auc']:.4f}")

    # all_results['alpha_beta'] = alpha_beta_results

    # 2. 调整特征维度 (初始嵌入维度)
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Initial Embedding Dimension")
    print("=" * 80)
    dimension_results = {}

    for dim in PARAM_GRIDS['dimension']['values']:
        config = DEFAULT_CONFIG.copy()
        config['EMBED_DIM'] = dim  # <--- 【关键修改】修改 embed_dim 而不是 contrastive_dim

        print(f"\n>>> Testing Embed Dimension={dim}")
        results = run_single_experiment(config, device)
        dimension_results[dim] = results

        print(f"  AUC: {results['mean']['auc']:.4f} ± {results['std']['auc']:.4f}")
        print(f"  AUPR: {results['mean']['pr_auc']:.4f} ± {results['std']['pr_auc']:.4f}")

    all_results['dimension'] = dimension_results

    # # 3. 调整训练Epochs
    # print("\n" + "=" * 80)
    # print("EXPERIMENT 3: Training Epochs")
    # print("=" * 80)
    # epochs_results = {}
    #
    # for num_epochs in PARAM_GRIDS['epochs']['values']:
    #     config = DEFAULT_CONFIG.copy()
    #     config['NUM_EPOCHS'] = num_epochs
    #
    #     print(f"\n>>> Testing Epochs={num_epochs}")
    #     results = run_single_experiment(config, device)
    #     epochs_results[num_epochs] = results
    #
    #     print(f"  AUC: {results['mean']['auc']:.4f} ± {results['std']['auc']:.4f}")
    #     print(f"  AUPR: {results['mean']['pr_auc']:.4f} ± {results['std']['pr_auc']:.4f}")
    #
    # all_results['epochs'] = epochs_results
    #
    # # 4. 调整卷积核
    # print("\n" + "=" * 80)
    # print("EXPERIMENT 4: Multi-Scale Convolution Kernels")
    # print("=" * 80)
    # kernels_results = {}
    #
    # for kernels in PARAM_GRIDS['conv_kernels']['values']:
    #     config = DEFAULT_CONFIG.copy()
    #     config['CONV_KERNELS'] = kernels
    #     kernel_str = ''.join(map(str, kernels))
    #
    #     print(f"\n>>> Testing Kernels={kernels}")
    #     results = run_single_experiment(config, device)
    #     kernels_results[kernel_str] = results
    #
    #     print(f"  AUC: {results['mean']['auc']:.4f} ± {results['std']['auc']:.4f}")
    #     print(f"  AUPR: {results['mean']['pr_auc']:.4f} ± {results['std']['pr_auc']:.4f}")
    #
    # all_results['conv_kernels'] = kernels_results
    #
    # # 保存所有结果
    # save_results(all_results, timestamp)
    #
    # return all_results


def save_results(results, timestamp):
    """保存调参结果到JSON和CSV文件"""
    # 保存完整JSON
    json_path = os.path.join(RESULTS_DIR, f"tuning_results_{timestamp}.json")
    with open(json_path, 'w') as f:
        # 转换numpy类型为Python原生类型
        serializable_results = convert_to_serializable(results)
        json.dump(serializable_results, f, indent=2)

    print(f"\n✅ Results saved to {json_path}")

    # 保存简化的CSV（每个实验一个文件）
    for exp_name, exp_results in results.items():
        csv_path = os.path.join(RESULTS_DIR, f"{exp_name}_{timestamp}.csv")
        with open(csv_path, 'w') as f:
            f.write("Parameter,AUC_Mean,AUC_Std,AUPR_Mean,AUPR_Std\n")
            for param, metrics in exp_results.items():
                f.write(f"{param},"
                        f"{metrics['mean']['auc']:.6f},"
                        f"{metrics['std']['auc']:.6f},"
                        f"{metrics['mean']['pr_auc']:.6f},"
                        f"{metrics['std']['pr_auc']:.6f}\n")
        print(f"✅ CSV saved to {csv_path}")


def convert_to_serializable(obj):
    """递归转换numpy类型为Python原生类型"""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


# ============================================================================
# 主程序入口
# ============================================================================
if __name__ == "__main__":
    cuda_name = "cuda:0"
    if len(sys.argv) > 1:
        cuda_name = "cuda:" + str(int(sys.argv[1]))

    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}\n')

    start_time = time.time()
    all_results = tune_hyperparameters(device)
    elapsed = time.time() - start_time

    print(f"\n{'=' * 80}")
    print(f"TUNING COMPLETED in {elapsed / 3600:.2f} hours")
    print(f"{'=' * 80}")