import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# 获取脚本所在的目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 切换到脚本所在目录
os.chdir(script_dir)

# 读取数据
train_df = pd.read_csv("train.csv")
test_df=pd.read_csv("test.csv")
train_df=train_df.drop(['Unnamed: 0','id'],axis=1)
test_df=test_df.drop(['Unnamed: 0','id'],axis=1)

# 填充缺失值,使用训练数据集中的“Arrival Delay in Minutes”（到达延误时间）这一列的中位数来填充缺失值
train_df['Arrival Delay in Minutes']=train_df['Arrival Delay in Minutes'].fillna(train_df['Arrival Delay in Minutes'].median())

# 类别编码
from sklearn.preprocessing import LabelEncoder  # 导入LabelEncoder类，用于将类别特征转换为数值型
encoder = LabelEncoder()  # 创建LabelEncoder实例
# 定义需要进行编码的类别特征列
category_features = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
# 遍历类别特征列，对每一列进行Label Encoding
for col in category_features:
    train_df[col] = encoder.fit_transform(train_df[col])  # 使用fit_transform对每个类别特征进行编码
# 对目标变量进行编码
y_encoder = LabelEncoder()
train_df['satisfaction']= y_encoder.fit_transform(train_df['satisfaction'])

# 数据集切分,将特征（X）和目标变量（y）分开
y_train_all = train_df['satisfaction']  # 目标变量：满意度
X_train_all = train_df.drop(columns=['satisfaction'])  # 特征数据：去除“satisfaction”列

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)
