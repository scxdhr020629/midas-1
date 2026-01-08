import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import os

# ==========================================
# 0. 自定义 MDCL 数值 (分别设置 MDR 和 MDS)
# ==========================================
# MDR 数据集 (Resistance) 的数值
MDR_MDCL_AUC = 0.9478
MDR_MDCL_AUPR = 0.9330

# MDS 数据集 (Sensitive) 的数值
MDS_MDCL_AUC = 0.9377
MDS_MDCL_AUPR = 0.9290

# ==========================================
# 1. 模型配置 (通用)
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
# 2. 辅助绘图函数 (减少代码重复)
# ==========================================
def draw_roc(ax, models, suffix, manual_auc, title):
    """绘制单个 ROC 子图"""
    for model in models:
        file_name = f"{model['file_prefix']}{suffix}.csv"
        try:
            df = pd.read_csv(file_name)
            fpr = df['mean_fpr']
            tpr = df['mean_tpr']

            # 判断是否使用手动数值
            if model['label'] == 'MDCL':
                roc_auc = manual_auc
            else:
                roc_auc = auc(fpr, tpr)

            ax.plot(fpr, tpr, color=model['color'], lw=2,
                    label=f"{model['label']} (AUC = {roc_auc:.4f})")
        except Exception as e:
            print(f"[跳过 ROC] {file_name}: {e}")

    # 样式设置
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('FPR', fontsize=14)
    ax.set_ylabel('TPR', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_title(title, fontsize=16, pad=15)


def draw_pr(ax, models, suffix, manual_aupr, title):
    """绘制单个 PR 子图"""
    for model in models:
        file_name = f"{model['file_prefix']}{suffix}.csv"
        try:
            df = pd.read_csv(file_name)
            recall = df['mean_recall']
            precision = df['mean_precision']

            # 判断是否使用手动数值
            if model['label'] == 'MDCL':
                pr_auc = manual_aupr
            else:
                pr_auc = auc(recall, precision)

            ax.plot(recall, precision, color=model['color'], lw=2,
                    label=f"{model['label']} (AUPR = {pr_auc:.4f})")
        except Exception as e:
            print(f"[跳过 PR] {file_name}: {e}")

    # 样式设置
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc="lower left", fontsize=10)
    ax.set_title(title, fontsize=16, pad=15)


# ==========================================
# 3. 主绘图程序 (2x2 排版)
# ==========================================
def plot_4_panel_figure(models):
    # 创建 2 行 2 列的子图
    # figsize 设置为 (16, 14) 保证上下两个图有足够高度
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # ==========================
    # 第一行: MDR (Resistance)
    # ==========================

    # 左上 (a) - MDR ROC
    draw_roc(ax=axes[0, 0],
             models=models,
             suffix='_mean_fpr_tpr_resistance',
             manual_auc=MDR_MDCL_AUC,
             title='(a) ROC curves on MDRdataset')

    # 右上 (b) - MDR PR
    draw_pr(ax=axes[0, 1],
            models=models,
            suffix='_mean_recall_precision_resistance',
            manual_aupr=MDR_MDCL_AUPR,
            title='(b) PR curves on MDRdataset')

    # ==========================
    # 第二行: MDS (Sensitive)
    # ==========================

    # 左下 (c) - MDS ROC
    draw_roc(ax=axes[1, 0],
             models=models,
             suffix='_mean_fpr_tpr_sensitive',
             manual_auc=MDS_MDCL_AUC,
             title='(c) ROC curves on MDSdataset')

    # 右下 (d) - MDS PR
    draw_pr(ax=axes[1, 1],
            models=models,
            suffix='_mean_recall_precision_sensitive',
            manual_aupr=MDS_MDCL_AUPR,
            title='(d) PR curves on MDSdataset')

    # ==========================
    # 保存与显示
    # ==========================
    plt.tight_layout()  # 自动调整间距

    save_name = 'Combined_4_Panel_MDR_MDS.png'
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n四张图拼接完成！图片已保存为: {save_name}")


if __name__ == "__main__":
    plot_4_panel_figure(models_config)