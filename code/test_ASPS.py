import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd  # 用于整理结果
import itertools     # 用于生成参数组合
from model_fixed import AttnFusionGCNNet
from utils import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from torch_geometric.loader import DataLoader

# ============================================================================
# 🛠️ 搜索空间配置 (在此处修改你想尝试的范围)
# ============================================================================
SEARCH_SPACE = {
    # 预热轮数：尝试 短期(3), 中期(5), 长期(8)
    'warmup_epochs': [3, 5, 8],
    
    # 坡度系数 (Ramp Rate)：决定 ASPS 多快达到完全状态
    # 0.4 = 较早完全激活 (激进)
    # 0.6 = 中庸
    # 0.8 = 较晚完全激活 (保守, 适合噪声大的数据)
    'ramp_rate': [0.4, 0.6, 0.8]
}

# 固定参数
NUM_EPOCHS = 35       # 稍微增加总轮数以容纳不同的坡度
NUM_FOLDS_FOR_TUNING = 5  # 调参时用几折交叉验证 (5折最准，3折较快)
DEVICE_ID = 0

# 其他固定超参数
LR = 0.0005
WEIGHT_DECAY = 0.0032
BATCH_SIZE = 64
CONTRASTIVE_DIM = 128
ALPHA, BETA, GAMMA = 0.5, 0.5, 1.0
TEMPERATURE = 0.1
LAM = 0.5
LOSS_FN = nn.BCELoss()

# ============================================================================
# 核心函数 (复用你的逻辑，但做了参数化改造)
# ============================================================================

def get_contrastive_weight(epoch, warmup_epochs):
    """根据传入的 warmup_epochs 动态计算权重"""
    if epoch <= warmup_epochs:
        progress = epoch / warmup_epochs
        return 0.5 * (1 - np.cos(np.pi * progress))
    return 1.0

def train_epoch(model, device, loader, optimizer, epoch, total_epochs, warmup_epochs):
    model.train()
    total_loss = 0
    batch_count = 0
    contrastive_weight_factor = get_contrastive_weight(epoch, warmup_epochs)

    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)

        output, loss_dict = model(
            data,
            current_epoch=epoch,
            total_epochs=total_epochs,
            warmup_epochs=warmup_epochs,
            return_contrastive_loss=True
        )

        labels = data.y.view(-1, 1).float().to(device)
        output = output.view(-1, 1)
        output = torch.clamp(output, min=1e-7, max=1.0 - 1e-7)
        
        loss_bce = LOSS_FN(output, labels)
        loss = (GAMMA * loss_bce +
                contrastive_weight_factor * (ALPHA * loss_dict['contrastive_mirna'] +
                                             BETA * loss_dict['contrastive_drug']))

        if torch.isnan(loss) or torch.isinf(loss):
            return None # 标记为失败

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        total_loss += loss.item()
        batch_count += 1

    return total_loss / batch_count

def evaluate(model, device, loader, total_epochs, warmup_epochs):
    model.eval()
    total_probs, total_labels = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(
                data, current_epoch=0, total_epochs=total_epochs, 
                warmup_epochs=warmup_epochs, return_contrastive_loss=False
            )
            total_probs.extend(output.cpu().numpy().flatten())
            total_labels.extend(data.y.view(-1, 1).cpu().numpy().flatten())
    
    total_probs = np.array(total_probs)
    total_labels = np.array(total_labels)
    try:
        return roc_auc_score(total_labels, total_probs)
    except:
        return 0.5

# ============================================================================
# 网格搜索主逻辑
# ============================================================================
def run_grid_search():
    cuda_name = f"cuda:{DEVICE_ID}"
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    print(f"🚀 开始 ASPS 参数网格搜索 on {device}...")
    print(f"搜索空间: {SEARCH_SPACE}\n")

    # 生成所有参数组合
    keys, values = zip(*SEARCH_SPACE.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    results = []

    for idx, params in enumerate(param_combinations):
        warmup = params['warmup_epochs']
        ramp = params['ramp_rate']
        
        # 动态计算全激活 Epoch
        full_activation = int(warmup + ramp * (NUM_EPOCHS - warmup))
        
        print(f"[{idx+1}/{len(param_combinations)}] Testing: Warmup={warmup}, Ramp={ramp} (Full Act at Epoch {full_activation})")
        
        fold_aucs = []
        
        # 运行交叉验证
        for fold in range(NUM_FOLDS_FOR_TUNING):
            # 加载数据
            train_data = TestbedDataset(root='data', dataset='train' + str(fold))
            test_data = TestbedDataset(root='data', dataset='test' + str(fold))
            train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
            test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

            # 初始化模型 (确保每次都是新的)
            model = AttnFusionGCNNet(
                n_output=1, n_filters=32, embed_dim=64, num_features_xd=78,
                num_features_smile=66, num_features_xt=25, output_dim=128, dropout=0.2,
                contrastive_dim=CONTRASTIVE_DIM, temperature=TEMPERATURE, lam=LAM
            ).to(device)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=LR * 0.01)

            # 训练循环
            model_failed = False
            for epoch in range(1, NUM_EPOCHS + 1):
                loss = train_epoch(model, device, train_loader, optimizer, epoch, NUM_EPOCHS, warmup)
                if loss is None:
                    print(f"   ⚠️ Fold {fold} NaN Loss detected. Skipping params.")
                    model_failed = True
                    break
                scheduler.step()
            
            if model_failed:
                fold_aucs.append(0.0)
            else:
                # 只取最后一轮结果
                auc_score = evaluate(model, device, test_loader, NUM_EPOCHS, warmup)
                fold_aucs.append(auc_score)
        
        # 记录该组参数的平均表现
        mean_auc = np.mean(fold_aucs)
        std_auc = np.std(fold_aucs)
        print(f"   👉 Result: Mean AUC = {mean_auc:.4f} ± {std_auc:.4f}\n")
        
        results.append({
            'warmup': warmup,
            'ramp_rate': ramp,
            'mean_auc': mean_auc,
            'std_auc': std_auc
        })

    # ============================================================================
    # 结果汇总与最佳选择
    # ============================================================================
    print("="*70)
    print("🏆 网格搜索完成！结果汇总：")
    print("="*70)
    
    # 转为 DataFrame 方便查看
    df = pd.DataFrame(results)
    df = df.sort_values(by='mean_auc', ascending=False) # 按 AUC 降序排列
    
    print(df.to_string(index=False))
    
    best_params = df.iloc[0]
    print("\n" + "="*70)
    print(f"✅ 最佳参数组合:")
    print(f"   Warmup Epochs: {int(best_params['warmup'])}")
    print(f"   Ramp Rate:     {best_params['ramp_rate']}")
    print(f"   Best Mean AUC: {best_params['mean_auc']:.5f}")
    print("="*70)

if __name__ == "__main__":
    run_grid_search()