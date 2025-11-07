import sys
import torch
import torch.nn as nn
import numpy as np
from model_1 import AttnFusionGCNNet  # <-- [修改] 导入新模型
from utils import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
from torch_geometric.data import DataLoader
import os
import optuna  # 导入 Optuna
import time

# --- 全局常量 ---
LOG_INTERVAL = 45
NUM_FOLDS = 5
loss_fn = nn.BCELoss()


# ============================================================================
#
# 核心训练/预测函数
#
# ============================================================================

def train(model, device, train_loader, optimizer, epoch):
    """训练一个epoch"""
    # print('Training on {} samples...'.format(len(train_loader.dataset))) # 在优化时太吵
    model.train()
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        output = model(data)
        labels = data.y.view(-1, 1).float().to(device)

        if torch.isnan(output).any():
            print(f"\n[!!! 致命错误：模型输出 NaN !!!]")
            raise ValueError("Model output is NaN. Stopping training.")

        epsilon = 1e-7
        output_clamped = torch.clamp(output, min=epsilon, max=1.0 - epsilon)

        if labels.min() < 0.0 or labels.max() > 1.0:
            print(f"\n[!!! 致命错误：标签越界 !!!]")
            raise ValueError("Labels out of bounds for BCELoss. Check your data.")

        loss = loss_fn(output_clamped, labels)

        if torch.isnan(loss):
            print(f"\n[!!! 致命错误：损失(Loss)为 NaN !!!]")
            raise ValueError("Loss is NaN. Stopping training.")

        loss.backward()
        optimizer.step()


def predicting(model, device, loader):
    """预测函数 - 只返回指标"""
    model.eval()
    total_probs = []
    total_labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            probs = output.cpu().numpy()
            total_probs.extend(probs)
            total_labels.extend(data.y.view(-1, 1).cpu().numpy())

    total_probs = np.array(total_probs).flatten()
    total_labels = np.array(total_labels).flatten()
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
#
# 5-Fold CV 训练函数
#
# ============================================================================

def run_cv_training(params, device, trial=None):  # <-- [修改] 移除 ablation_mode
    """
    运行完整的 5-Fold 交叉验证 (无绘图)
    """

    # 从 params 字典中解包超参数
    LR = params["lr"]
    NUM_EPOCHS = params["num_epochs"]
    TRAIN_BATCH_SIZE = params["batch_size"]
    TEST_BATCH_SIZE = params["batch_size"]
    WEIGHT_DECAY = params["weight_decay"]

    # TODO: 从 params 中解包你模型的超参数
    # 例如:
    # HIDDEN_DIM = params["hidden_dim"]
    # DROPOUT_RATE = params["dropout_rate"]

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    roc_aucs = []
    pr_aucs = []

    modeling = AttnFusionGCNNet  # <-- [修改] 使用新模型

    for fold in range(NUM_FOLDS):
        train_data = TestbedDataset(root='data', dataset='train' + str(fold))
        test_data = TestbedDataset(root='data', dataset='test' + str(fold))
        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, drop_last=True)

        # 初始化模型
        # TODO: 将模型的超参数传入构造函数
        model = modeling(
            # hidden_dim=HIDDEN_DIM,
            # dropout_rate=DROPOUT_RATE
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        for epoch in range(NUM_EPOCHS):
            train(model, device, train_loader, optimizer, epoch + 1)

        accuracy, precision, recall, f1, roc_auc, pr_auc = predicting(model, device, test_loader)

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        roc_aucs.append(roc_auc)
        pr_aucs.append(pr_auc)

        if trial:
            trial.report(pr_auc, fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    avg_pr_auc = np.mean(pr_aucs)

    return avg_pr_auc


# ============================================================================
#
# Optuna Objective Function
#
# ============================================================================

def objective(trial, device):  # <-- [修改] 移除 ablation_mode
    """Optuna 的 Objective 函数"""

    # 1. 定义超参数的搜索空间
    params = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "num_epochs": trial.suggest_int("num_epochs", 20, 70)

        # TODO: 在这里添加你的模型超参数
        # "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
        # "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
        # "num_layers": trial.suggest_int("num_layers", 2, 5),
    }

    # 2. 运行 5-Fold CV 并获取
    try:
        avg_pr_auc = run_cv_training(
            params=params,
            device=device,
            trial=trial
        )
        return avg_pr_auc

    except optuna.exceptions.TrialPruned as e:
        raise e
    except Exception as e:
        print(f"[!!! Trial 失败 !!!] ID: {trial.number}, Params: {trial.params}")
        print(f"  Error: {e}")
        # 在堆栈跟踪中打印更详细的错误
        import traceback
        traceback.print_exc()
        return -1.0

    # ============================================================================


#
# Main Execution
#
# ============================================================================

if __name__ == "__main__":

    # --- 1. 基本设置 ---
    cuda_name = "cuda:0"
    if len(sys.argv) > 1:
        cuda_name = "cuda:" + str(int(sys.argv[1]))
    print('cuda_name:', cuda_name)

    # (移除了 ablation_mode)
    # ablation_mode = sys.argv[2] if len(sys.argv) > 2 else 'baseline'
    # print(f"Ablation mode: {ablation_mode}")

    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}\n')

    # --- 2. Optuna Study 设置 ---

    # [修改] 简化了 study 名称
    storage_name = f"sqlite:///optuna_study.db"
    study_name = f"optimization_attn_model"

    print(f"Optuna study storage: {storage_name}")
    print(f"Optuna study name: {study_name}")

    study = optuna.create_study(
        direction="maximize",
        storage=storage_name,
        study_name=study_name,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1)
    )

    # --- 3. 运行优化 ---
    N_TRIALS = 50

    print(f"\nStarting Optuna optimization for '{study_name}'...")
    print(f"Running {N_TRIALS} new trials (Total trials will be {len(study.trials) + N_TRIALS}).")

    start_time = time.time()
    study.optimize(
        lambda trial: objective(trial, device),  # <-- [修改] 移除 ablation_mode
        n_trials=N_TRIALS,
        timeout=60 * 60 * 4
    )
    end_time = time.time()

    print(f"\nOptimization finished in {(end_time - start_time) / 60:.2f} minutes.")

    # --- 4. 打印优化结果 ---
    print("\n" + "=" * 70)
    print("Optimization Summary")
    print("=" * 70)
    print(f"Number of finished trials: {len(study.trials)}")

    try:
        best_trial = study.best_trial
        print("\nBest trial:")
        print(f"  Value (Max Avg PR-AUC): {best_trial.value:.6f}")

        print("\n  Best hyperparameters:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
    except ValueError:
        print("\n[!!! 错误 !!!] 未找到最佳试验。")
        print("这可能意味着所有试验都失败了（返回 -1.0）。")
        print("请检查上面打印的 'Trial 失败' 错误日志。")

    # --- 5. 最终步骤 (移除了绘图) ---
    print("\n" + "=" * 70)
    print(f"🎉 Bayesian optimization complete!")
    print(f"   Optuna study saved to '{storage_name}'")