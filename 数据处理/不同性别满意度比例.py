import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
train_df = pd.read_csv("./train.csv")
train_df = train_df.drop(['Unnamed: 0', 'id'], axis=1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))# 创建一个包含两个子图的图形，设置图形大小为15x7
colors = ['#FF9999', '#66B2FF']# 设置饼图的颜色方案

male_data = train_df[train_df['Gender'] == 'Male']['satisfaction'].value_counts()# 获取男性乘客的满意度数据
male_percent = (male_data / len(train_df[train_df['Gender'] == 'Male']) * 100).round(1)# 计算男性乘客满意度的百分比
female_data = train_df[train_df['Gender'] == 'Female']['satisfaction'].value_counts()# 获取女性乘客的满意度数据
female_percent = (female_data / len(train_df[train_df['Gender'] == 'Female']) * 100).round(1)#计算女性乘客满意度的百分比
# 绘制男性乘客满意度饼图
wedges1, texts1, autotexts1 = ax1.pie(male_data,                               
    labels=[f'{label}\n({value}%)' for label, value in zip(male_data.index, male_percent)],
    colors=colors,autopct='',
    wedgeprops={'width': 0.7, 'edgecolor': 'white', 'linewidth': 2})
ax1.set_title('男性乘客满意度分布', fontsize=12, pad=20)# 设置男性乘客饼图的标题
# 绘制女性乘客满意度饼图
wedges2, texts2, autotexts2 = ax2.pie(female_data,
    labels=[f'{label}\n({value}%)' for label, value in zip(female_data.index, female_percent)],
    colors=colors,autopct='',
    wedgeprops={'width': 0.7, 'edgecolor': 'white', 'linewidth': 2})
ax2.set_title('女性乘客满意度分布', fontsize=12, pad=20)

plt.suptitle('不同性别乘客满意度比例分布', fontsize=14, y=1.05)# 设置整个图表的标题
plt.figlegend(wedges1, male_data.index, title='满意度类型',# 添加图例，显示满意度类型 
              loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()# 自动调整子图之间的间距
plt.show()
print("\n男性乘客满意度分布：")
for label, value in zip(male_data.index, male_percent):
    print(f"{label}: {value}%")
print("\n女性乘客满意度分布：")
for label, value in zip(female_data.index, female_percent):
    print(f"{label}: {value}%")
