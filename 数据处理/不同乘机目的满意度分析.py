import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
train_df = pd.read_csv("./train.csv")
train_df = train_df.drop(['Unnamed: 0', 'id'], axis=1)

travel_satisfaction = pd.crosstab(train_df['Type of Travel'], train_df['satisfaction'])
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
sns.countplot(data=train_df, x="Type of Travel", ax=axes[0], palette=['#B8E0D2'])
axes[0].tick_params(axis='x', rotation=0)
axes[0].set_ylabel('数量')
axes[0].set_xlabel('乘机目的')
axes[0].set_title('不同乘机目的的样本数量分布', fontsize=16)
for i in axes[0].containers:
    axes[0].bar_label(i, padding=3, fmt='%d')
travel_satisfaction.plot(kind='bar', stacked=True, ax=axes[1], 
                        color=['#FF9999', '#66B2FF'])
axes[1].tick_params(axis='x', rotation=0)
axes[1].set_ylabel('数量')
axes[1].set_xlabel('乘机目的')
axes[1].set_title('不同乘机目的的满意度分析', fontsize=16)
for c in axes[1].containers:
    axes[1].bar_label(c, label_type='center')
axes[1].legend(title='满意度')
plt.tight_layout()
plt.show()
print("\n不同乘机目的满意度分布：")
print("\n绝对数量：")
print(travel_satisfaction)

print("\n百分比分布：")
percentage = travel_satisfaction.div(travel_satisfaction.sum(axis=1), axis=0) * 100
print(percentage.round(2))

satisfaction_ratio = (travel_satisfaction['satisfied'] / travel_satisfaction.sum(axis=1) * 100).round(2)
print("\n各乘机目的满意比例：")
for travel_type, ratio in satisfaction_ratio.items():
    print(f"{travel_type}: {ratio}%")
