import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
# 获取脚本所在的目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 切换到脚本所在目录
os.chdir(script_dir)

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

train_df=pd.read_csv("train.csv")

# 创建一个 2x2 的子图布局，设置整个图形大小为 (15, 8) 英寸
f, ax = plt.subplots(2, 2, figsize = (15,8))
# 绘制第一个子图：展示 "Inflight entertainment"（机上娱乐）与 "Flight Distance"（飞行距离）的箱型图
sns.boxplot(data=train_df, 
            x="Inflight entertainment",    # x 轴：机上娱乐
            y="Flight Distance",           # y 轴：飞行距离
            hue="Inflight entertainment",  # 按照机上娱乐类型对数据进行分组
            palette="YlOrBr",              # 使用黄色到橙色渐变的调色板
            legend=False,                  # 不显示图例
            ax=ax[0, 0])                   # 将该图放在 2x2 子图的第一行第一列（0,0）位置
# 绘制第二个子图：展示 "Flight Distance" 与 "Inflight entertainment" 的堆叠直方图
sns.histplot(train_df, 
             x="Flight Distance",               # x 轴：飞行距离
             hue="Inflight entertainment",      # 根据机上娱乐类型堆叠不同颜色
             multiple="stack",                 # 堆叠显示多个类别
             palette="YlOrBr",                  # 使用黄色到橙色渐变的调色板
             edgecolor=".3",                    # 设置条形边缘颜色为灰色
             linewidth=.5,                      # 设置条形边缘线宽为 0.5
             ax=ax[0, 1])                       # 将该图放在 2x2 子图的第一行第二列（0,1）位置
# 绘制第三个子图：展示 "Leg room service"（腿部空间服务）与 "Flight Distance" 的箱型图
sns.boxplot(data=train_df, 
            x="Leg room service",     # x 轴：腿部空间服务
            y="Flight Distance",      # y 轴：飞行距离
            hue="Leg room service",   # 按照腿部空间服务类型对数据进行分组
            palette="YlOrBr",         # 使用黄色到橙色渐变的调色板
            legend=False,             # 不显示图例
            ax=ax[1, 0])              # 将该图放在 2x2 子图的第二行第一列（1,0）位置

# 绘制第四个子图：展示 "Flight Distance" 与 "Leg room service" 的堆叠直方图
sns.histplot(train_df, 
             x="Flight Distance",        # x 轴：飞行距离
             hue="Leg room service",     # 根据腿部空间服务类型堆叠不同颜色
             multiple="stack",           # 堆叠显示多个类别
             palette="YlOrBr",            # 使用黄色到橙色渐变的调色板
             edgecolor=".3",              # 设置条形边缘颜色为灰色
             linewidth=.5,                # 设置条形边缘线宽为 0.5
             ax=ax[1, 1])                 # 将该图放在 2x2 子图的第二行第二列（1,1）位置
plt.show()# 显示所有图形

