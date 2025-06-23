import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
train_df = pd.read_csv("./train.csv")
train_df = train_df.drop(['Unnamed: 0', 'id'], axis=1)
# 创建客户类型和满意度的交叉表
customer_type = pd.crosstab(train_df['Customer Type'], train_df['satisfaction'])
# 创建包含两个子图的图形，设置图形大小为20x6
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
# 绘制客户类型数量分布柱状图
sns.countplot(data=train_df, x="Customer Type", ax=axes[0], palette=['#FF9999', '#66B2FF'])

axes[0].tick_params(axis='x', rotation=0)# 设置x轴标签不旋转
axes[0].set_ylabel('数量')# 设置y轴标签
axes[0].set_xlabel('客户类型')# 设置x轴标签
axes[0].set_title('不同类型客户的样本数量分布', fontsize=16)# 设置第一个子图的标题
for i in axes[0].containers:# 为柱状图添加数值标签
    axes[0].bar_label(i, padding=3, fmt='%d')
customer_type.plot(kind='bar', stacked=True, ax=axes[1], # 绘制堆叠柱状图显示满意度分布
                  color=['#FF9999', '#66B2FF'],
                  fontsize=12)
axes[1].tick_params(axis='x', rotation=0)# 设置x轴标签不旋转
axes[1].set_ylabel('数量')# 设置y轴标签
axes[1].set_xlabel('客户类型')# 设置x轴标签
axes[1].set_title('不同类型客户的满意度分析', fontsize=16)# 设置第二个子图的标题
for c in axes[1].containers:# 为堆叠柱状图添加数值标签
    axes[1].bar_label(c, label_type='center')
axes[1].legend(title='满意度')# 添加图例标题
plt.tight_layout()# 自动调整子图之间的间距
plt.show()# 显示图表

# 打印不同类型客户满意度分布
print("\n不同类型客户满意度分布：")
print("\n绝对数量：")
print(customer_type)
print("\n百分比分布：")
percentage = customer_type.div(customer_type.sum(axis=1), axis=0) * 100
print(percentage.round(2))