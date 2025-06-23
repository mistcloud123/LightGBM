import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
train_df = pd.read_csv("./train.csv")
train_df = train_df.drop(['Unnamed: 0', 'id'], axis=1)

train_df['Age_Group'] = pd.cut(train_df['Age'], 
                              bins=[0, 20, 30, 40, 50, 60, 100],
                              labels=['0-20岁', '21-30岁', '31-40岁', '41-50岁', '51-60岁', '60岁以上'])
age_satisfaction = pd.crosstab(train_df['Age_Group'], train_df['satisfaction'])

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
sns.countplot(data=train_df, x="Age_Group", ax=axes[0], palette=['#A8D8B9'])
axes[0].tick_params(axis='x', rotation=45)
axes[0].set_ylabel('数量')
axes[0].set_xlabel('年龄组')
axes[0].set_title('不同年龄组的样本数量分布', fontsize=16)
for i in axes[0].containers:
    axes[0].bar_label(i, padding=3, fmt='%d')
age_satisfaction.plot(kind='bar', stacked=True, ax=axes[1], 
                     color=['#FF9999', '#66B2FF'])
axes[1].tick_params(axis='x', rotation=45)
axes[1].set_ylabel('数量')
axes[1].set_xlabel('年龄组')
axes[1].set_title('不同年龄组的满意度分析', fontsize=16)
for c in axes[1].containers:
    axes[1].bar_label(c, label_type='center')
axes[1].legend(title='满意度')
plt.tight_layout()
plt.show()
print("\n不同年龄组满意度分布：")
print("\n绝对数量：")
print(age_satisfaction)
print("\n百分比分布：")
percentage = age_satisfaction.div(age_satisfaction.sum(axis=1), axis=0) * 100
print(percentage.round(2))
satisfaction_ratio = (age_satisfaction['satisfied'] / age_satisfaction.sum(axis=1) * 100).round(2)
print("\n各年龄组满意比例：")
for age_group, ratio in satisfaction_ratio.items():
    print(f"{age_group}: {ratio}%")
