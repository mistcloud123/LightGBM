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

# 计算每个评分下满意和不满意的人数
satisfaction_counts = pd.crosstab(train_df['Online boarding'], train_df['satisfaction'])
all_ratings = pd.DataFrame(index=range(6))# 确保包含所有评分（0-5）
satisfaction_counts = satisfaction_counts.reindex(all_ratings.index).fillna(0)
satisfaction_pct = satisfaction_counts.div(satisfaction_counts.sum(axis=1), axis=0) * 100# 计算百分比
plt.figure(figsize=(12, 8))# 创建图表
plt.plot(satisfaction_pct.index, satisfaction_pct['neutral or dissatisfied'], # 画折线图
         marker='o', color='#1f77b4', linewidth=2, label='neutral or dissatisfied')
plt.plot(satisfaction_pct.index, satisfaction_pct['satisfied'], 
         marker='o', color='#ff7f0e', linewidth=2, label='satisfied')
plt.title('在线登机满意度与整体满意度的分布情况', fontsize=14, pad=15)
plt.xlabel('在线登机评分', fontsize=12)
plt.ylabel('percentage(%)', fontsize=12)
plt.legend(fontsize=10)
plt.xticks(range(6))# 设置x轴刻度
plt.ylim(0, 100)# 设置y轴范围
# 添加数值标签
for i in satisfaction_pct.index:
    if not np.isnan(satisfaction_pct.loc[i, 'neutral or dissatisfied']):
        plt.text(i, satisfaction_pct.loc[i, 'neutral or dissatisfied'], 
                f'{satisfaction_pct.loc[i, "neutral or dissatisfied"]:.1f}%', 
                ha='center', va='bottom')
        plt.text(i, satisfaction_pct.loc[i, 'satisfied'], 
                f'{satisfaction_pct.loc[i, "satisfied"]:.1f}%', 
                ha='center', va='bottom')

plt.tight_layout()
plt.show()

# 打印各评分下的满意度分布
print("\n各在线登机评分的满意度分布：")
print("\n评分人数：")
print(satisfaction_counts)
print("\n满意度百分比：")
satisfaction_pct.columns = ['neutral or dissatisfied', 'satisfied']
print(satisfaction_pct.round(2))
