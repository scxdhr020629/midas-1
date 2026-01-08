import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 数据准备 (Data Preparation)
# ==========================================

# 实验 A: Lambda (λ)
data_lambda = {
    'x': [0.1, 0.3, 0.5, 0.7, 0.9],
    'auc': [0.9453, 0.9457, 0.9478, 0.9466, 0.9466],
    'aupr': [0.9287, 0.9289, 0.9330, 0.9302, 0.9323],
    'xlabel': 'λ',  # 支持 LaTeX 格式
    'filename': 'Lambda_metrics.png'          # 指定文件名
}

# 实验 B: Embedding Dimension
data_dim = {
    'x': ['32', '64', '128', '256', '512'], 
    'auc': [0.9448, 0.9456, 0.9478, 0.9443, 0.9471],
    'aupr': [0.9298, 0.9299, 0.9330, 0.9285, 0.9332],
    'xlabel': 'Embedding Dimension',
    'filename': 'Dimension_metrics.png'
}

# 实验 C: Epoch
data_epoch = {
    'x': [10, 20, 30, 40, 50],
    'auc': [0.9433, 0.9447, 0.9478, 0.9460, 0.9467],
    'aupr': [0.9265, 0.9284, 0.9330, 0.9287, 0.9310],
    'xlabel': 'Epochs',
    'filename': 'Epoch_metrics.png'
}

# 实验 D: Convolution Kernel Combinations
data_kernel = {
    'x': ['1,2,3', '2,3,4', '3,4,5', '4,5,6'],
    'auc': [0.9461, 0.9478, 0.9463, 0.9468],
    'aupr': [0.9311, 0.9330, 0.9311, 0.9313],
    'xlabel': 'Kernel Combinations',
    'filename': 'Kernel_metrics.png'
}

# 将所有数据放入列表，方便循环处理
all_experiments = [data_lambda, data_dim, data_epoch, data_kernel]

# ==========================================
# 2. 绘图设置 (Plot Configuration)
# ==========================================

# 设置全局字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 14  # 单张图字体可以稍微大一点
# '#9DD9F3', '#C1C2E1'
# 定义颜色
# color_auc = '#F5B70A'  # yellow
# color_aupr = '#C0D8A8' # green
color_auc = '#9DD9F3'  # 蓝色
color_aupr = '#C1C2E1' # 紫色
# ==========================================
# 3. 定义单张图绘制函数
# ==========================================
def plot_and_save_single_chart(data):
    """
    绘制并保存单张图表
    """
    x = data['x']
    auc = data['auc']
    aupr = data['aupr']
    xlabel = data['xlabel']
    filename = data['filename']
    
    # 创建独立的画布 (figsize 可以根据需要调整，单张图通常 8x6 或 6x5 比较合适)
    fig, ax = plt.subplots(figsize=(8, 6))

    # 如果 x 是字符串列表（Dimension 或 Kernel），我们需要生成索引来保证等间距
    is_categorical = isinstance(x[0], str)
    if is_categorical:
        x_indices = range(len(x))
    else:
        x_indices = x

    # 绘制 AUC 线 (红色，圆点)
    ax.plot(x_indices, auc, marker='o', color=color_auc, label='AUC', 
            linewidth=2.5, markersize=9, alpha=0.9)
    
    # 绘制 AUPR 线 (蓝色，方块)
    ax.plot(x_indices, aupr, marker='s', color=color_aupr, label='AUPR', 
            linewidth=2.5, markersize=9, alpha=0.9)

    # 设置坐标轴标签
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel('Values', fontsize=14, fontweight='bold')

    # 处理 X 轴刻度
    if is_categorical:
        ax.set_xticks(x_indices)
        ax.set_xticklabels(x)
    else:
        ax.set_xticks(x)
    
    # 自动调整 Y 轴范围 (让波动看起来更明显，上下留出余量)
    all_values = auc + aupr
    y_min, y_max = min(all_values), max(all_values)
    margin = (y_max - y_min) * 0.25 # 稍微增加一点边距
    ax.set_ylim(y_min - margin, y_max + margin)

    # 添加图例
    ax.legend(loc='best', frameon=True, fontsize=12)
    
    # 设置边框粗细
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        
    # 去除网格
    ax.grid(False) 

    # 保存图片逻辑
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    # 打印提示
    print(f"已生成并保存图片: {filename}")
    
    # 关闭画布，释放内存（重要，否则循环多了会卡）
    plt.close(fig)

# ==========================================
# 4. 执行循环生成
# ==========================================

print("开始生成图片...")
for experiment_data in all_experiments:
    plot_and_save_single_chart(experiment_data)
print("所有图片生成完毕。")