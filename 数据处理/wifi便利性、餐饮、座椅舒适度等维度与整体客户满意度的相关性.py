import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 获取脚本所在的目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 切换到脚本所在目录
os.chdir(script_dir)

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

train_df=pd.read_csv("train.csv")

train_df = train_df.iloc[:, 8:]#删除前8列，只保留需要讨论的属性列
# 将 satisfaction 列转换为二值形式：satisfied=1，其他=0
train_df['satisfaction'] = (train_df['satisfaction'] == 'satisfied').astype(int)
correlation_matrix = train_df.corr()# 计算相关性矩阵
# 创建热力图,设定大小为 12x10 英寸
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, 
            annot=True,         # 在每个方格中显示数字
            cmap='Reds',        # 使用红色渐变色调
            vmin=0.0,           # 设置色图的最小值为 0.0
            vmax=0.5,           # 设置色图的最大值为 0.5
            center=0.25,        # 设置色图的中心值为 0.25
            fmt='.2f')          # 设置显示的小数精度为 2 位
plt.title('特征相关性热力图')# 设置图表的标题
plt.xticks(rotation=45, ha='right')# 设置 x 轴的刻度标签旋转 45 度并将其对齐到右侧
plt.yticks(rotation=0)# 设置 y 轴的刻度标签旋转 0 度（保持水平）
plt.tight_layout()# 自动调整布局，防止标签和标题重叠
plt.show()# 显示图表

print("\n与satisfaction最相关的特征（按相关性绝对值排序）：")
# 计算与 "satisfaction" 特征的相关性，并按相关性的绝对值进行排序
correlations_with_satisfaction = correlation_matrix['satisfaction'].sort_values(key=abs, ascending=False)
print(correlations_with_satisfaction)

