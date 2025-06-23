import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                            recall_score, f1_score, confusion_matrix,
                            classification_report, roc_curve, precision_recall_curve)

# 读取数据
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

train_df = train_df.drop(['Unnamed: 0', 'id'], axis=1)
test_df = test_df.drop(['Unnamed: 0', 'id'], axis=1)

# 缺失值填充
train_df['Arrival Delay in Minutes'] = train_df['Arrival Delay in Minutes'].fillna(
    train_df['Arrival Delay in Minutes'].median())

# 类别编码
encoder = LabelEncoder()
category_features = ['Gender', 'Customer Type', 'Type of Travel', 'Class']

for col in category_features:
    train_df[col] = encoder.fit_transform(train_df[col])

# 对目标变量进行编码
y_encoder = LabelEncoder()
train_df['satisfaction'] = y_encoder.fit_transform(train_df['satisfaction'])

# 数据集切分
y_train_all = train_df['satisfaction']
X_train_all = train_df.drop(columns=['satisfaction'])

X_train, X_valid, y_train, y_valid = train_test_split(X_train_all, y_train_all, test_size=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

print('train: {}, valid: {}, test: {}'.format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))

df_columns = X_train.columns.values
print('feature count: {}'.format(len(df_columns)))

# 检查训练集和验证集的标签分布
print()
print("Train labels:", pd.Series(y_train).value_counts(normalize=True))
print("Valid labels:", pd.Series(y_valid).value_counts(normalize=True))

# 检查训练集和验证集是否有重叠索引（如有）
train_indices = set(X_train.index)
valid_indices = set(X_valid.index)
print("重叠样本数:", len(train_indices & valid_indices))


# ==================== LightGBM建模与优化 ====================

# 设置LightGBM参数
lgb_params = {
    'boosting_type': 'gbdt',          # 传统梯度提升决策树
    'objective': 'binary',            # 二分类任务
    'metric': 'auc',                  # 评估指标为AUC
    'learning_rate': 0.1,             # 学习率
    'num_leaves': 31,                 # 叶子节点数
    'max_depth': -1,                  # 树的最大深度，-1表示不限制
    'min_child_samples': 20,          # 叶子节点最小样本数
    'subsample': 0.8,                 # 数据采样比例
    'colsample_bytree': 0.8,          # 特征采样比例
    'reg_alpha': 0.0,                 # L1正则化
    'reg_lambda': 0.0,                # L2正则化
    'random_state': 42,               # 随机种子
    'n_jobs': -1,                     # 使用所有CPU核心
    'verbose': -1                     # 不输出日志信息
}

# 创建LightGBM数据集
dtrain = lgb.Dataset(X_train, label=y_train, feature_name=list(df_columns))
dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain, feature_name=list(df_columns))

# 训练模型
print(f"\n\n=== model training ===")
model = lgb.train(
    params=lgb_params,
    train_set=dtrain,
    valid_sets=[dtrain, dvalid],
    valid_names=['train', 'valid'],
    num_boost_round=1000,
    callbacks=[
        lgb.log_evaluation(50),       # 每50轮打印一次日志
        lgb.early_stopping(100)       # 100轮无提升则停止
    ]
)


# 模型评估
train_pred = model.predict(X_train)
valid_pred = model.predict(X_valid)
test_pred = model.predict(X_test)

# 将概率转换为类别预测
train_pred_class = (train_pred > 0.5).astype(int)
valid_pred_class = (valid_pred > 0.5).astype(int)
test_pred_class = (test_pred > 0.5).astype(int)

# 计算评估指标
train_accuracy = accuracy_score(y_train, train_pred_class)
valid_accuracy = accuracy_score(y_valid, valid_pred_class)
test_accuracy = accuracy_score(y_test, test_pred_class)

train_auc = roc_auc_score(y_train, train_pred)
valid_auc = roc_auc_score(y_valid, valid_pred)
test_auc = roc_auc_score(y_test, test_pred)

# 打印评估结果
print("\n=== Model Performance ===")
print(f"Train Accuracy: {train_accuracy:.4f}, AUC: {train_auc:.4f}")
print(f"Valid Accuracy: {valid_accuracy:.4f}, AUC: {valid_auc:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}, AUC: {test_auc:.4f}")

# 特征重要性
feature_importance = pd.DataFrame({
    'Feature': df_columns,
    'Importance': model.feature_importance()
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
plt.title('Top 20 Feature Importance')
plt.tight_layout()
plt.show()


# ==================== 模型评估与可视化 ====================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False # 显示负号

# ==================== 模型性能评估 ====================

# 预测训练集
predict_train = model.predict(X_train)
train_auc = roc_auc_score(y_train, predict_train)

# 预测验证集
predict_valid = model.predict(X_valid)
valid_auc = roc_auc_score(y_valid, predict_valid)

# 预测测试集
predict_test = model.predict(X_test)
test_auc = roc_auc_score(y_test, predict_test)


def evaluate_binary(y_true, y_pred_prob, dataset_name=""):
    """二分类评估函数"""
    # 获取预测类别（默认阈值0.5）
    y_pred = (y_pred_prob > 0.5).astype(int)

    # 计算评估指标
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_prob)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n=== {dataset_name} 评估结果 ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # 分类报告
    print("\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=y_encoder.classes_))

    # 绘制ROC曲线
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{dataset_name} ROC曲线')
    plt.legend()

    # 绘制PR曲线
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_prob)
    plt.subplot(1, 2, 2)
    plt.plot(recall_curve, precision_curve, label=f'F1 = {f1:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{dataset_name} PR曲线')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 混淆矩阵
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=y_encoder.classes_,
                yticklabels=y_encoder.classes_)
    plt.title(f'{dataset_name} 混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.show()

    return accuracy, auc


# 在训练集、验证集和测试集上进行评估
print("\n" + "=" * 50)
train_pred = model.predict(X_train)
train_acc, train_auc = evaluate_binary(y_train, train_pred, "训练集")
valid_pred = model.predict(X_valid)
valid_acc, valid_auc = evaluate_binary(y_valid, valid_pred, "验证集")
test_pred = model.predict(X_test)
test_acc, test_auc = evaluate_binary(y_test, test_pred, "测试集")
# 打印整体性能
print("\n" + "=" * 50)
print("=== 模型整体性能 ===")
print(f"训练集: Accuracy = {train_acc:.4f}, AUC = {train_auc:.4f}")
print(f"验证集: Accuracy = {valid_acc:.4f}, AUC = {valid_auc:.4f}")
print(f"测试集: Accuracy = {test_acc:.4f}, AUC = {test_auc:.4f}")

# ==================== 特征重要性分析 ====================
# 获取特征重要性（按重要性增益排序）
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model.feature_importance(importance_type='gain')
}).sort_values('Importance', ascending=False)

# 可视化Top20特征
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
plt.title('Top 20 特征重要性 (Gain)')
plt.tight_layout()
plt.show()

# 保存重要特征
top_features = feature_importance.head(20)['Feature'].values
print("\nTop 20重要特征:", top_features)

# 输出最优迭代轮数
print(f"\n最优迭代轮数: {model.best_iteration}")