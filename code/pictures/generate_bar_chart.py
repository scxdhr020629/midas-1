import matplotlib.pyplot as plt

# ==========================================
# 0. 全局设置 (两个图通用的风格)
# ==========================================
plt.rcParams['font.sans-serif'] = ['Arial']  # 设置字体
plt.rcParams['axes.unicode_minus'] = False
# 这个是绿色得使用
# bar_color = '#AAE8E3'  # 统一颜色
bar_color = '#9AD8FF'  # 统一颜色
# ==========================================
# 1. 第一个图：miRNA 序列长度分布
# ==========================================

# 准备数据
labels_left = ['<20', '20', '21', '22', '23', '24', '>24']
values_left = [114, 77, 213, 582, 199, 44, 23]

# 创建独立画布 (figsize可以根据需要单独调整)
plt.figure(figsize=(7, 6))

# 绘制柱状图
bars1 = plt.bar(labels_left, values_left, color=bar_color, width=0.6)

# 设置标签
plt.xlabel('Length of miRNA sequence (chars)', fontsize=12)
plt.ylabel('Count', fontsize=12)

# 设置Y轴范围
plt.ylim(0, max(values_left) * 1.1)

# 添加数值标签
for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'{int(height)}',
             ha='center', va='bottom', fontsize=11, color='black')

# 保存并显示 (bbox_inches='tight' 防止边缘显示不全)
plt.tight_layout()
plt.savefig('miRNA_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
print("已保存图片: miRNA_distribution.png")


# ==========================================
# 2. 第二个图：Drug SMILES 长度分布
# ==========================================

# 准备数据
labels_right = ['<20', '20-40', '40-60', '60-80', '80-100', '100-120', '>120']
values_right = [6, 13, 40, 40, 16, 7, 18]

# 创建新的独立画布
plt.figure(figsize=(7, 6))

# 绘制柱状图
bars2 = plt.bar(labels_right, values_right, color=bar_color, width=0.6)

# 设置标签
plt.xlabel('Length of drug SMILES (chars)', fontsize=12)
plt.ylabel('Count', fontsize=12)

# 设置Y轴范围
plt.ylim(0, max(values_right) * 1.1)

# 添加数值标签
for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{int(height)}',
             ha='center', va='bottom', fontsize=11, color='black')

# 保存并显示
plt.tight_layout()
plt.savefig('SMILES_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
print("已保存图片: SMILES_distribution.png")