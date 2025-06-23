import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
train_df = pd.read_csv("./train.csv")
train_df = train_df.drop(['Unnamed: 0', 'id'], axis=1)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
sns.kdeplot(data=train_df[train_df['satisfaction'] == 'neutral or dissatisfied'],
            x='Flight Distance',
            label='不满意',
            ax=ax1,
            color='#FF9999')
sns.kdeplot(data=train_df[train_df['satisfaction'] == 'satisfied'],
            x='Flight Distance',
            label='满意',
            ax=ax1,
            color='#66B2FF')
ax1.set_title('不同航程距离的满意度分布', fontsize=16)
ax1.set_xlabel('航程距离')
ax1.set_ylabel('密度')
ax1.legend(title='满意度')
sns.boxplot(data=train_df, x='satisfaction', y='Flight Distance',
           ax=ax2,
           palette=['#FF9999', '#66B2FF'])
ax2.set_title('不同满意度的航程距离分布', fontsize=16)
ax2.set_xlabel('满意度')
ax2.set_ylabel('航程距离')
plt.tight_layout()
plt.show()

print("\n航程距离统计信息：")
print("\n按满意度分组的航程距离描述性统计：")
print(train_df.groupby('satisfaction')['Flight Distance'].describe().round(2))
distance_bins = [0, 500, 1000, 2000, 4000, float('inf')]
distance_labels = ['0-500', '501-1000', '1001-2000', '2001-4000', '4000+']
train_df['Distance_Group'] = pd.cut(train_df['Flight Distance'], 
                                  bins=distance_bins, 
                                  labels=distance_labels)

distance_satisfaction = pd.crosstab(train_df['Distance_Group'], train_df['satisfaction'])
distance_ratio = (distance_satisfaction['satisfied'] / distance_satisfaction.sum(axis=1) * 100).round(2)
print("\n不同距离段的满意度比例：")
for distance, ratio in distance_ratio.items():
    print(f"{distance}公里: {ratio}%满意")
