import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 分组标签和策略名称
labels = ['0.01 s', '0.001 s', '0.0001 s', '0.00001 s']
strategies = ['Base NCCL', 'Reconf. WAN', 'Reconf. CCL', 'Reconf. WAN + CCL']

# 示例数据：4组，每组4个策略值
data = np.array([
    [38326.03, 36144.67, 35621.24, 34959.75],
    [13092.11, 10858.57, 9554.54, 8119.77],
    [11696.90, 8063.98, 6734.49, 5788.01],
    [10345.42, 7768.10, 6482.19, 3419.14],
])
data = data.T  # 转置以适应绘图逻辑

# 人民币配色
colors = ['#EEB969', '#8AA173', '#8FB1CF', '#B9BAD7']

# 坐标计算
x = np.arange(len(labels))
bar_width = 0.2
offsets = np.linspace(-1.5, 1.5, len(strategies)) * bar_width

# 绘图
plt.figure(figsize=(8, 2.5))

for i in range(len(strategies)):
    plt.bar(x + offsets[i], data[i], width=bar_width, color=colors[i], label=strategies[i])

# 设置字体大小
font_size = 17
plt.xticks(x, labels, fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.ylabel('Completion Time', fontsize=font_size)

# 使用科学计数法显示纵轴
ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax.yaxis.get_offset_text().set_fontsize(font_size)  # 设置 ×10^x 的字体大小

# 设置图例
plt.legend(
    frameon=False,
    fontsize=font_size,
    ncol=2,
    columnspacing=0.2,
    handletextpad=0.1,
    handlelength=1.5,
    handleheight=0.8
)

plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()