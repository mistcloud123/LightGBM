import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

train_df = pd.read_csv("./train.csv")
train_df = train_df.drop(['Unnamed: 0', 'id'], axis=1)

# 创建分类图，展示不同因素对满意度的影响
g = sns.catplot(
    x="Flight Distance", # 设置x轴为航程距离
    y="Type of Travel", # 设置y轴为旅行类型
    hue="satisfaction", # 使用满意度作为颜色区分
    col="Class", # 按舱位等级分列显示
    data=train_df, # 使用训练数据集
    kind="bar", # 使用柱状图类型
    height=6, # 设置图形高度
    aspect=1.2, # 设置图形宽高比
    palette=['#FF9999', '#66B2FF'] # 设置颜色方案
)

# 设置图表总标题
plt.suptitle('不同航班舱位、旅行类型和航程距离对满意度影响', fontsize=16)
g.set_axis_labels('航程距离', '旅行类型')# 设置x轴和y轴的标签
plt.subplots_adjust(wspace=0.2, top=0.85)# 调整子图之间的间距和顶部边距
plt.show()# 显示图表

print("\n各舱位的满意度统计：")
for class_type in train_df['Class'].unique():
    class_data = train_df[train_df['Class'] == class_type]
    satisfaction_ratio = (class_data['satisfaction'] == 'satisfied').mean() * 100
    print(f"\n{class_type}舱：")
    print(f"满意度比例: {satisfaction_ratio:.2f}%")
    print("按旅行类型细分：")
    for travel_type in class_data['Type of Travel'].unique():
        travel_data = class_data[class_data['Type of Travel'] == travel_type]
        travel_satisfaction = (travel_data['satisfaction'] == 'satisfied').mean() * 100
        print(f"{travel_type}: {travel_satisfaction:.2f}%满意")
