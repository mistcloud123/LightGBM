import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
train_df = pd.read_csv("./train.csv")
train_df = train_df.drop(['Unnamed: 0', 'id'], axis=1)

f, ax = plt.subplots(1, 2, figsize=(15, 5))
sns.boxplot(x="Customer Type", y="Age", 
           palette="YlOrBr", 
           data=train_df, 
           ax=ax[0])
ax[0].set_title('不同类型乘客的年龄分布箱线图', fontsize=16)
ax[0].set_xlabel('客户类型')
ax[0].set_ylabel('年龄')
sns.histplot(data=train_df, 
            x="Age", 
            hue="Customer Type", 
            multiple="stack", 
            palette="YlOrBr", 
            edgecolor=".3", 
            linewidth=.5, 
            ax=ax[1])
ax[1].set_title('不同类型乘客的年龄分布密度直方图', fontsize=16)
ax[1].set_xlabel('年龄')
ax[1].set_ylabel('数量')
plt.tight_layout()
plt.show()
print("\n不同类型乘客的年龄统计：")
age_stats = train_df.groupby('Customer Type')['Age'].describe()
print(age_stats)
age_bins = [0, 20, 30, 40, 50, 60, 100]
age_labels = ['0-20岁', '21-30岁', '31-40岁', '41-50岁', '51-60岁', '60岁以上']
train_df['Age_Group'] = pd.cut(train_df['Age'], bins=age_bins, labels=age_labels)
age_customer_dist = pd.crosstab(train_df['Age_Group'], train_df['Customer Type'], normalize='index') * 100
print("\n各年龄段的客户类型分布（百分比）：")
print(age_customer_dist.round(2))
print("\n不同类型客户的满意度：")
satisfaction_by_type = pd.crosstab(train_df['Customer Type'], train_df['satisfaction'])
satisfaction_ratio = (satisfaction_by_type['satisfied'] / satisfaction_by_type.sum(axis=1) * 100).round(2)
for customer_type, ratio in satisfaction_ratio.items():
    print(f"{customer_type}: {ratio}%满意")