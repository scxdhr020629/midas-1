import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 全局数据与配置
# ==========================================
group_labels = ['AUC', 'AUPR']
model_names = ['MDCL', 'w/o-SS', 'w/o-CF', 'w/o-CA', 'w/o-CCL']
colors = ['#9DD9F3', '#C1C2E1', '#C0D8A8', '#EDD2E5', '#C6C6C6']

# --- MDR 数据 (左图) ---
mdr_auc = [0.9478, 0.9385, 0.9421, 0.9386, 0.9373]
mdr_aupr = [0.9330, 0.9192, 0.9254, 0.9226, 0.9191]

# --- MDS 数据 (右图) ---
mds_auc = [0.9377, 0.9267, 0.9320, 0.9307, 0.9280]
mds_aupr = [0.9290, 0.9141, 0.9216, 0.9213, 0.9182]


# ==========================================
# 2. 绘图辅助函数
# ==========================================
def draw_subplot(ax, auc_vals, aupr_vals, ylim_range, label_text):
    """
    绘制单个子图的函数
    """
    x = np.arange(len(group_labels))
    total_width = 0.8
    n_bars = len(model_names)
    bar_width = total_width / n_bars

    # 循环绘制柱子
    for i in range(n_bars):
        x_pos = x - (total_width / 2) + (i * bar_width) + (bar_width / 2)
        scores = [auc_vals[i], aupr_vals[i]]

        rects = ax.bar(x_pos, scores,
                       width=bar_width,
                       label=model_names[i],
                       color=colors[i],
                       edgecolor='white',
                       linewidth=0.5)

        # 添加数值标签
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height + 0.0003,
                    f'{height:.4f}',
                    ha='center', va='bottom',
                    fontsize=8,
                    color='black')

    # --- 样式设置 ---
    ax.set_ylim(ylim_range)  # 设置Y轴范围

    # 坐标轴标签
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=14)  # 不加粗
    ax.tick_params(axis='y', labelsize=10)

    # 图例 (右上角，框内)
    ax.legend(loc='upper right',
              ncol=1,
              fontsize=9,
              edgecolor='#D3D3D3',
              framealpha=0.9,
              borderaxespad=1)

    # 网格线
    ax.yaxis.grid(True, linestyle='--', which='major', color='gray', alpha=0.3)
    ax.set_axisbelow(True)

    # 边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#333333')
        spine.set_linewidth(0.8)

    # --- 关键：在下方添加 (a)/(b) 标注 ---
    # 已去掉 fontweight='bold'
    ax.set_xlabel(label_text, fontsize=16, labelpad=15)


# ==========================================
# 3. 主绘图逻辑
# ==========================================

# 创建画布：宽20，高6
fig, axes = plt.subplots(1, 2, figsize=(20, 6), dpi=300)

# --- 绘制左图 (a) MDR ---
draw_subplot(ax=axes[0],
             auc_vals=mdr_auc,
             aupr_vals=mdr_aupr,
             ylim_range=(0.915, 0.955),
             label_text='(a) MDRdataset')

# --- 绘制右图 (b) MDS ---
draw_subplot(ax=axes[1],
             auc_vals=mds_auc,
             aupr_vals=mds_aupr,
             ylim_range=(0.91, 0.95),
             label_text='(b) MDSdataset')

# 调整整体布局
plt.tight_layout()

# 保存图片
plt.savefig('ablation_study_combined_final.png', dpi=300, bbox_inches='tight')
plt.show()

print("图表已生成：ablation_study_combined_final.png")