import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 数据准备
# ==========================================
# 两个主组
group_labels = ['AUC', 'AUPR']

# 7个变体名称
model_names = ['MDCL', 'w/o MS', 'w/o MC', 'w/o DS', 'w/o DF', 'w/o CA', 'w/o CCL']

# 数值数据
auc_data = [0.9478, 0.9387, 0.9462, 0.9442, 0.9439, 0.9467, 0.9440]
aupr_data = [0.9330, 0.9199, 0.9310, 0.9273, 0.9273, 0.9299, 0.9295]

# 指定颜色
colors = ['#D080A8', '#E0A0C0', '#9DD9F3', '#C1C2E1', '#C0D8A8', '#EDD2E5', '#C6C6C6']

# ==========================================
# 2. 绘图参数设置
# ==========================================
x = np.arange(len(group_labels))  # X轴位置 [0, 1]
total_width = 0.8                 # 一组柱子的总宽度
n_bars = len(model_names)         # 7个柱子
bar_width = total_width / n_bars  # 单个柱子的宽度

# 创建画布，设置宽一点，方便横向排版
fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

# ==========================================
# 3. 循环绘制柱子 & 添加数值标签
# ==========================================
for i in range(n_bars):
    # 计算当前柱子的X坐标：
    # x - (总宽/2) + (偏移量)
    x_pos = x - (total_width / 2) + (i * bar_width) + (bar_width / 2)
    
    # 提取当前模型在 AUC 和 AUPR 的分数
    scores = [auc_data[i], aupr_data[i]]
    
    # 绘制柱子
    rects = ax.bar(x_pos, scores, 
                   width=bar_width, 
                   label=model_names[i], 
                   color=colors[i], 
                   edgecolor='white', 
                   linewidth=0.5)
    
    # --- 核心仿照点：在柱子上方添加数值 ---
    for rect in rects:
        height = rect.get_height()
        # 垂直显示数值，加上一点padding，字体设小一点以防重叠
        ax.text(rect.get_x() + rect.get_width() / 2, height + 0.0005,
                f'{height:.4f}', 
                ha='center', va='bottom', 
                fontsize=7.5,  # 字体大小，可根据需要调整
                color='black') # 字体颜色

# ==========================================
# 4. 样式美化 (仿照参考图)
# ==========================================

# --- Y轴范围设置 (关键) ---
# 参考数据最低 0.9199，最高 0.9478
# 设置为 0.90 ~ 0.96 可以很好地拉开差距
plt.ylim(0.91, 0.96)

# --- 坐标轴标签 ---
ax.set_xticks(x)
ax.set_xticklabels(group_labels, fontsize=14, fontweight='bold')
# ax.set_ylabel('Metrics', fontsize=12) # 如果需要Y轴文字可取消注释

# --- 图例设置 (关键：横着排) ---
# ncol=4 表示分4列排列，这就实现了“横着排”的效果
# frameon=True 加上边框，看起来更像论文插图
ax.legend(loc='upper center', 
          bbox_to_anchor=(0.5, 1.0), # 将图例放在顶部中间略偏下的位置
          ncol=4,                    # 分4列显示 (7个变体排成2行)
          fontsize=9, 
          edgecolor='gray', 
          framealpha=1)

# --- 网格线 ---
# 参考图背景很干净，这里只加淡灰色的横线
ax.yaxis.grid(True, linestyle='--', which='major', color='gray', alpha=0.3)
ax.set_axisbelow(True) # 让网格线在柱子下面

# --- 去掉上、右边框 ---
ax.spines['top'].set_visible(True)  # 参考图其实保留了边框，这里设为True形成封闭框
ax.spines['right'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('ablation_study_chart.png', dpi=300, bbox_inches='tight')
plt.show()