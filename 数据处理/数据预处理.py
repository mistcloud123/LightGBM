import pandas as pd# 导入pandas库，用于数据处理和分析
import numpy as np# 导入numpy库，用于数值计算
import matplotlib.pyplot as plt# 导入matplotlib.pyplot库，用于数据可视化
import seaborn as sns# 导入seaborn库，用于统计数据可视化

plt.rcParams['font.sans-serif'] = ['SimHei']# 设置matplotlib的中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False# 解决负号显示问题

train_df = pd.read_csv("./train.csv")# 读取训练数据集
test_df = pd.read_csv("./test.csv")# 读取测试数据集
 
# 删除训练数据集和测试数据集中不需要的列（'Unnamed: 0'和'id'列）
train_df = train_df.drop(['Unnamed: 0', 'id'], axis=1)
test_df = test_df.drop(['Unnamed: 0', 'id'], axis=1)

train_df.info()# 显示训练数据集的基本信息

satisfaction_counts = train_df['satisfaction'].value_counts()# 统计满意度分布情况
values = satisfaction_counts.values# 获取满意度计数值
labels = satisfaction_counts.index# 获取满意度标签
percent = (satisfaction_counts / len(train_df) * 100).round(1)# 计算满意度百分比

fig = plt.figure(figsize=(10,8))# 创建图形，设置图形大小为10x8
wedges, texts = plt.pie(values, wedgeprops={"width": 0.4, 'edgecolor': '#000', 'linewidth': 3},
                       colors=['#FF9999', '#66B2FF'])# 绘制环形图，设置环形图宽度和边框样式
kw = dict(arrowprops=dict(arrowstyle="-"), zorder=0, va="center")# 设置标注箭头的样式
 
for i, p in enumerate(wedges):# 为每个扇形添加标注
    ang = (p.theta2 - p.theta1) / 1.8 + p.theta1 # 计算标注位置的角度
    y = np.sin(np.deg2rad(ang)) # 计算标注的y坐标
    x = np.cos(np.deg2rad(ang))# 计算标注的x坐标
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))] # 根据x坐标确定文本对齐方式
    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    kw["arrowprops"].update({"connectionstyle": connectionstyle})# 设置连接线的样式
    plt.annotate( # 添加标注文本
        f"{labels[i]}\n{percent[i]}%",
        xy=(x, y),
        xytext=(1.35 * np.sign(x), 1.4 * y),
        horizontalalignment=horizontalalignment,
        fontsize=12,
        **kw
    )
plt.title("航空客户满意度情况占比", fontsize=16, pad=20)# 设置图表标题
plt.show()# 显示图表

# 打印详细的满意度分布情况
print("\n满意度分布情况：")
for label, value, pct in zip(labels, values, percent):
    print(f"{label}: {value} 人 ({pct}%)")
