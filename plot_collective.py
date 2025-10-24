import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 分组标签和策略名称
labels = ['8 MB', '16 MB', '32 MB', '64 MB']
strategies = ['Base NCCL', 'Reconf. CCL', 'Reconf. WAN', 'Reconf. WAN + CCL']

# 示例数据：4组，每组4个策略值
data = np.array([[1716.3365232448095, 1559.889865540594, 1532.8194395791056, 1268.8262238211757], [3386.5216876462864, 3077.718183722371, 3015.7109082775346, 2493.0361495556617], [6726.995980374869, 6114.1048187814295, 5981.428827618787, 4941.459276325452], [13407.991019370782, 12186.710433202417, 11912.830856908442, 9838.126365205302]])
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

# 使用科学记数法显示纵轴
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
    handleheight=0.8,
    bbox_to_anchor=(0.75, 0.49)
)

plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
