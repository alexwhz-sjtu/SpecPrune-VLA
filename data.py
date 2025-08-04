import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据
tokens = [512-32,512-64,512-128,512-256,512-384,512-512]
accuracy = [53.3,86.7,90,93.3,93.3,94.0]

# 创建图形和轴
fig, ax = plt.subplots(figsize=(12, 4))



# 绘制折线图
line = ax.plot(tokens, accuracy, linestyle='--', marker='s', linewidth=4, markersize=10, 
               color="#639CD8", 
               markeredgewidth=2.5, zorder=3)

# 设置图表属性
ax.set_xlabel('pruned token number', fontsize=20, fontweight='bold', labelpad=10)
ax.set_ylabel('task success rate (%)', fontsize=20, fontweight='bold', labelpad=10)

# 设置网格
ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.8, color="#585959", axis='y')

# 设置坐标轴
ax.set_xticks(tokens)
# ax.set_yticks(range(80, 101, 5))
ax.set_xlim(-10, 512)
ax.set_ylim(40, 102)

# 自定义刻度标签
ax.tick_params(axis='both', which='major', labelsize=20)



# 调整布局
plt.tight_layout()

# 显示图形
plt.show()

# 保存图片（可选）
plt.savefig('enhanced_token_pruning_accuracy.png', dpi=300, bbox_inches='tight')