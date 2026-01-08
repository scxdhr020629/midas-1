import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. MDR 数据准备 (Blue/Purple)
# ==========================================
mdr_lambda = {
    'x': [0.1, 0.3, 0.5, 0.7, 0.9],
    'auc': [0.9453, 0.9457, 0.9478, 0.9466, 0.9466],
    'aupr': [0.9287, 0.9289, 0.9330, 0.9302, 0.9323],
    'label': '(a) Contrastive loss weight $\lambda$ on MDRdataset'
}

mdr_kernel = {
    'x': ['1,2,3', '2,3,4', '3,4,5', '4,5,6'],
    'auc': [0.9461, 0.9478, 0.9463, 0.9468],
    'aupr': [0.9311, 0.9330, 0.9311, 0.9313],
    'label': '(b) Convolution kernel $k$ combinations on MDRdataset'
}

mdr_dim = {
    'x': ['32', '64', '128', '256', '512'],
    'auc': [0.9448, 0.9456, 0.9478, 0.9443, 0.9471],
    'aupr': [0.9298, 0.9299, 0.9330, 0.9285, 0.9332],
    'label': '(c) Embedding dimension $d_m$ ($d_r$) on MDRdataset'
}

mdr_epoch = {
    'x': [10, 20, 30, 40, 50],
    'auc': [0.9433, 0.9447, 0.9478, 0.9460, 0.9467],
    'aupr': [0.9265, 0.9284, 0.9330, 0.9287, 0.9310],
    'label': '(d) epochs $epoch$ on MDRdataset'
}

# ==========================================
# 2. MDS 数据准备 (Yellow/Green)
# ==========================================
mds_lambda = {
    'x': [0.1, 0.3, 0.5, 0.7, 0.9],
    'auc': [0.9357, 0.9369, 0.9377, 0.9339, 0.9362],
    'aupr': [0.9252, 0.9264, 0.9290, 0.9241, 0.9277],
    'label': '(e) Contrastive loss weight $\lambda$ on MDSdataset'
}

mds_kernel = {
    'x': ['1,2,3', '2,3,4', '3,4,5', '4,5,6'],
    'auc': [0.9331, 0.9377, 0.9369, 0.9336],
    'aupr': [0.9210, 0.9290, 0.9288, 0.9233],
    'label': '(f) Convolution kernel $k$ combinations on MDSdataset'
}

mds_dim = {
    'x': ['32', '64', '128', '256', '512'],
    'auc': [0.9359, 0.9377, 0.9371, 0.9367, 0.9354],
    'aupr': [0.9290, 0.9290, 0.9267, 0.9286, 0.9214],
    'label': '(g) Embedding dimension $d_m$ ($d_r$) on MDSdataset'
}

mds_epoch = {
    'x': [10, 20, 30, 40, 50],
    'auc': [0.9330, 0.9369, 0.9377, 0.9358, 0.9372],
    'aupr': [0.9242, 0.9283, 0.9290, 0.9263, 0.9284],
    'label': '(h) epochs $epoch$ on MDSdataset'
}

all_data = [
    mdr_lambda, mdr_kernel,
    mdr_dim, mdr_epoch,
    mds_lambda, mds_kernel,
    mds_dim, mds_epoch
]

# ==========================================
# 3. 颜色设置
# ==========================================
color_auc_mdr = '#9DD9F3'  # 蓝色
color_aupr_mdr = '#C1C2E1'  # 紫色
color_auc_mds = '#F5B70A'  # 黄色
color_aupr_mds = '#C0D8A8'  # 绿色


# ==========================================
# 4. 绘图主程序
# ==========================================
def plot_combined_8_panel():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['font.size'] = 12

    # 创建 4行 x 2列 的画布
    # 修改：高度从 24 减小到 22，减少纵向空白
    fig, axes = plt.subplots(4, 2, figsize=(16, 22))

    axes_flat = axes.flatten()

    for i, ax in enumerate(axes_flat):
        data = all_data[i]

        if i < 4:
            c_auc = color_auc_mdr
            c_aupr = color_aupr_mdr
        else:
            c_auc = color_auc_mds
            c_aupr = color_aupr_mds

        x_raw = data['x']
        auc_val = data['auc']
        aupr_val = data['aupr']

        is_categorical = isinstance(x_raw[0], str)
        if is_categorical:
            x_indices = range(len(x_raw))
        else:
            x_indices = x_raw

        ax.plot(x_indices, auc_val, marker='o', color=c_auc, label='AUC',
                linewidth=2.5, markersize=8, alpha=0.9)
        ax.plot(x_indices, aupr_val, marker='s', color=c_aupr, label='AUPR',
                linewidth=2.5, markersize=8, alpha=0.9)

        if is_categorical:
            ax.set_xticks(x_indices)
            ax.set_xticklabels(x_raw, fontsize=11)
        else:
            ax.set_xticks(x_raw)
            ax.tick_params(axis='x', labelsize=11)

        # 纵坐标旁边的 Values 去掉加粗
        ax.set_ylabel('Values', fontsize=12, fontweight='normal')
        ax.tick_params(axis='y', labelsize=11)

        all_vals = auc_val + aupr_val
        y_min, y_max = min(all_vals), max(all_vals)
        margin = (y_max - y_min) * 0.35
        ax.set_ylim(y_min - margin, y_max + margin)

        ax.legend(loc='best', frameon=True, fontsize=10)

        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

        # --- 修改重点：标题位置 ---
        # 1. 移除了 ax.set_xlabel()
        # 2. 将标题的 y 从 -0.25 改为 -0.18，让标题贴近 X 轴，不再“漂浮”
        # 3. fontweight='normal' 不加粗
        ax.set_title(data['label'], y=-0.18, fontsize=14, fontweight='normal')

    plt.tight_layout()

    # --- 修改重点：调整子图间距 ---
    # hspace: 行间距，从 0.4 改为 0.25 (紧凑上下行)
    # wspace: 列间距，从 0.2 改为 0.15 (紧凑左右列)
    plt.subplots_adjust(hspace=0.25, wspace=0.15)

    save_name = 'Parameter_Sensitivity_8_Panel_Compact.png'
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"图表已生成并保存为: {save_name}")


if __name__ == "__main__":
    plot_combined_8_panel()