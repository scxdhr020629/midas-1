import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import os

# ==========================================
# 1. 模型配置
# ==========================================
# 列表包含每个模型的信息：
# 'file_prefix': 文件名前缀 (程序会自动拼接 _mean_fpr_tpr_resistance.csv)
# 'label': 图例中显示的名字
# 'color': 曲线颜色
models_config = [
    # --- 您的模型 ---
    {'file_prefix': 'MDCL',       'label': 'MDCL',       'color': '#d62728'}, # 红色
    
    # --- 对比模型 ---
    {'file_prefix': 'GraphDTA',   'label': 'GraphDTA',   'color': '#1f77b4'}, # 蓝色
    {'file_prefix': 'MDA',        'label': 'DLST_MDA',   'color': '#ff7f0e'}, # 橙色 (注意：这里标签改为 DLST_MDA)
    {'file_prefix': 'ML-DTI',     'label': 'ML_DTI',     'color': '#2ca02c'}, # 绿色
    {'file_prefix': 'SMIR_GCNN',  'label': 'GCNNMMA',    'color': '#9467bd'}, # 紫色 （标签改为 GCNNMMA）
    {'file_prefix': 'SubMDTA',    'label': 'SubMDTA',    'color': '#8c564b'}, # 棕色
]

# 设置绘图风格
plt.style.use('default') 
# plt.style.use('seaborn-whitegrid') # 可选风格

# ==========================================
# 2. 绘制 ROC 曲线函数
# ==========================================
def plot_all_roc_curves(models):
    plt.figure(figsize=(8, 8))
    
    # 遍历每个模型进行绘图
    for model in models:
        file_name = f"{model['file_prefix']}_mean_fpr_tpr_resistance.csv"
        
        try:
            # 读取数据
            df = pd.read_csv(file_name)
            fpr = df['mean_fpr']
            tpr = df['mean_tpr']
            
            # 计算 AUC
            roc_auc = auc(fpr, tpr)
            
            # 绘图
            plt.plot(fpr, tpr, color=model['color'], lw=2, 
                     label=f"{model['label']} (AUC = {roc_auc:.4f})")
            print(f"[ROC] 已加载: {model['label']} (AUC: {roc_auc:.4f})")
            
        except FileNotFoundError:
            print(f"[警告] 找不到文件: {file_name}，已跳过该模型。")
        except KeyError:
            print(f"[错误] 文件 {file_name} 列名不对，请检查是否包含 'mean_fpr' 和 'mean_tpr'。")

    # 绘制对角线 (随机猜测线)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # 设置图形属性
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('ROC Curve - Resistance Prediction', fontsize=14)
    
    # 图例位置：右下角
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 保存
    save_name = 'Combined_ROC_Curves.png'
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"所有模型的 ROC 曲线已保存为: {save_name}\n")

# ==========================================
# 3. 绘制 PR 曲线函数
# ==========================================
def plot_all_pr_curves(models):
    plt.figure(figsize=(8, 8))
    
    # 遍历每个模型进行绘图
    for model in models:
        file_name = f"{model['file_prefix']}_mean_recall_precision_resistance.csv"
        
        try:
            # 读取数据
            df = pd.read_csv(file_name)
            recall = df['mean_recall']
            precision = df['mean_precision']
            
            # 计算 AUPR
            pr_auc = auc(recall, precision)
            
            # 绘图
            plt.plot(recall, precision, color=model['color'], lw=2, 
                     label=f"{model['label']} (AUPR = {pr_auc:.4f})")
            print(f"[PR]  已加载: {model['label']} (AUPR: {pr_auc:.4f})")
            
        except FileNotFoundError:
            print(f"[警告] 找不到文件: {file_name}，已跳过该模型。")
        except KeyError:
            print(f"[错误] 文件 {file_name} 列名不对，请检查是否包含 'mean_recall' 和 'mean_precision'。")

    # 设置图形属性
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensitivity)', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve - Resistance Prediction', fontsize=14)
    
    # 图例位置：左下角
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 保存
    save_name = 'Combined_PR_Curves.png'
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"所有模型的 PR 曲线已保存为: {save_name}\n")

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    print("开始绘制多模型对比曲线...\n")
    
    # 绘制 ROC
    plot_all_roc_curves(models_config)
    
    # 绘制 PR
    plot_all_pr_curves(models_config)