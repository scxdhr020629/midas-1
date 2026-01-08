import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import os

# ==========================================
# 0. 自定义 MDCL 数值
# ==========================================
MDCL_MANUAL_AUC = 0.9478
MDCL_MANUAL_AUPR = 0.9330

# ==========================================
# 1. 模型配置
# ==========================================
models_config = [
    {'file_prefix': 'MDCL', 'label': 'MDCL', 'color': '#d62728'},
    {'file_prefix': 'GraphDTA', 'label': 'GraphDTA', 'color': '#1f77b4'},
    {'file_prefix': 'MDA', 'label': 'DLST_MDA', 'color': '#ff7f0e'},
    {'file_prefix': 'ML-DTI', 'label': 'ML_DTI', 'color': '#2ca02c'},
    {'file_prefix': 'SMIR_GCNN', 'label': 'GCNNMMA', 'color': '#9467bd'},
    {'file_prefix': 'SubMDTA', 'label': 'SubMDTA', 'color': '#8c564b'},
]

# 设置绘图风格
plt.style.use('default')


# ==========================================
# 2. 主绘图程序
# ==========================================
def plot_combined_figure(models):
    # 创建 1 行 2 列的子图，设置总大小 (宽16, 高7)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # -------------------------------------------------------
    # 子图 1: ROC Curves (左边: axes[0])
    # -------------------------------------------------------
    ax_roc = axes[0]

    for model in models:
        file_name = f"{model['file_prefix']}_mean_fpr_tpr_resistance.csv"
        try:
            df = pd.read_csv(file_name)
            fpr = df['mean_fpr']
            tpr = df['mean_tpr']

            # MDCL 使用手动值，其他自动计算
            if model['label'] == 'MDCL':
                roc_auc = MDCL_MANUAL_AUC
            else:
                roc_auc = auc(fpr, tpr)

            ax_roc.plot(fpr, tpr, color=model['color'], lw=2,
                        label=f"{model['label']} (AUC = {roc_auc:.4f})")
        except Exception as e:
            print(f"[跳过 ROC] {model['label']}: {e}")

    # ROC 样式设置
    ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('FPR', fontsize=14)
    ax_roc.set_ylabel('TPR', fontsize=14)
    ax_roc.grid(True, linestyle='--', alpha=0.6)
    ax_roc.legend(loc="lower right", fontsize=10)

    # *** 设置左图标题 ***
    ax_roc.set_title('(a) ROC curves on MDRdataset', fontsize=16, pad=15)

    # -------------------------------------------------------
    # 子图 2: PR Curves (右边: axes[1])
    # -------------------------------------------------------
    ax_pr = axes[1]

    for model in models:
        file_name = f"{model['file_prefix']}_mean_recall_precision_resistance.csv"
        try:
            df = pd.read_csv(file_name)
            recall = df['mean_recall']
            precision = df['mean_precision']

            # MDCL 使用手动值，其他自动计算
            if model['label'] == 'MDCL':
                pr_auc = MDCL_MANUAL_AUPR
            else:
                pr_auc = auc(recall, precision)

            ax_pr.plot(recall, precision, color=model['color'], lw=2,
                       label=f"{model['label']} (AUPR = {pr_auc:.4f})")
        except Exception as e:
            print(f"[跳过 PR] {model['label']}: {e}")

    # PR 样式设置
    ax_pr.set_xlim([0.0, 1.0])
    ax_pr.set_ylim([0.0, 1.05])
    ax_pr.set_xlabel('Recall', fontsize=14)
    ax_pr.set_ylabel('Precision', fontsize=14)
    ax_pr.grid(True, linestyle='--', alpha=0.6)
    ax_pr.legend(loc="lower left", fontsize=10)

    # *** 设置右图标题 ***
    # 注意：这里修正了您输入中的 typos "MDSdataseet" -> "MDSdataset"
    # 如果您确实需要保留那个拼写，请将下方字符串改回。
    ax_pr.set_title('(b) PR curves on MDRdataset', fontsize=16, pad=15)

    # -------------------------------------------------------
    # 保存与显示
    # -------------------------------------------------------
    plt.tight_layout()  # 自动调整间距，防止重叠

    save_name = 'Combined_ROC_PR_SideBySide_MDR.png'
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n拼接完成！图片已保存为: {save_name}")


if __name__ == "__main__":
    plot_combined_figure(models_config)