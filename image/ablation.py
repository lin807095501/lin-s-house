import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rcParams

config = {
            "font.family": 'serif',
            "font.size": 13,# 相当于小四大小
            "mathtext.fontset": 'stix',#matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
            "font.serif": ['Times New Roman'],#宋体
            'axes.unicode_minus': False # 处理负号，即-号
         }
rcParams.update(config)

size =4
# 返回size个0-1的随机数
SA = np.array([85.58, 85.83,84.56,84.75])
FGSM =  np.array([60.85, 64.42, 62.48,62.72])
PGD10 = np.array([56.4, 56.72, 59.03,59.38])
PGD50 = np.array([55.93, 56.38, 58.88,59.12])
AA = np.array([51.37,51.54,53.25,53.79])
# x轴坐标, size=2, 返回[0, 1, 2]
x = np.arange(size)

# 有a/b/c三种类型的数据，n设置为3
total_width, n = 0.8, 5
# 每种类型的柱状图宽度
width = total_width / n 

# 重新设置x轴的坐标
x = x - (total_width - width) / 2 

# 设置xy坐标名称
# plt.xlabel("Datasets")
plt.ylabel("Accuracy")

# 设置y坐标取值范围
plt.ylim([50, 90])


# 画柱状图
plt.bar(x - width, SA, width=width, label="SA", edgecolor='black', linewidth=0.5, color="#F38181", hatch='/')
plt.bar(x, FGSM, width=width, label="FGSM", edgecolor='black', linewidth=0.5, color="#FCE38A", hatch='-')
plt.bar(x + width, PGD10, width=width, label="PGD10", edgecolor='black', linewidth=0.5, color="#EAFFD0", hatch='\\')
plt.bar(x + 2 * width, PGD50, width=width, label="PGD50", edgecolor='black', linewidth=0.5, color="#95E1D3", hatch='X')
plt.bar(x + 3 * width, AA, width=width, label="AA", edgecolor='black', linewidth=0.5, color="#F7CBC5", hatch='\/')
# 显示图例
plt.legend(loc=2)

# 功能1
x_labels = ["L_instacne", "L_cluster", "L_strength","Total"]
# 用第1组...替换横坐标x的值
plt.xticks(x, x_labels)
plt.tick_params(labelsize=14)

# 功能2
for i, j in zip(x - width, SA):
    plt.text(i, j + 0.01, "%.1f" % j, ha="center", va="bottom")
for i, j in zip(x, FGSM):
    plt.text(i, j + 0.01, "%.1f" % j, ha="center", va="bottom")
for i, j in zip(x + width, PGD10):
    plt.text(i, j + 0.01, "%.1f" % j, ha="center", va="bottom")
for i, j in zip(x + 2 * width, PGD50):
    plt.text(i, j + 0.01, "%.1f" % j, ha="center", va="bottom")
for i, j in zip(x + 2 * width, AA):
    plt.text(i, j + 0.01, "%.1f" % j, ha="center", va="bottom")

# 显示柱状图
plt.savefig(fname="/data/lsb/image/pdf/ablation_loss.pdf", dpi=300)