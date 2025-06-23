import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, precision_score, recall_score, roc_auc_score, make_scorer, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# 获取脚本所在的目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

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

# 特征和标签
y = train_df['satisfaction']
X = train_df.drop(columns=['satisfaction'])

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 自定义评分函数
f1_scorer = make_scorer(f1_score, average='binary')

# ========== 定义模型评估函数 ==========
def get_model_metrics_and_plot(model, X, y, model_name):
    t0 = time.time()
    model.fit(X, y)
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    train_score = model.score(X, y)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, pos_label=1)
    recall = recall_score(y, y_pred, pos_label=1)
    roc = roc_auc_score(y, y_pred_proba)
    time_taken = time.time() - t0

    print(f"{model_name} - Accuracy = {accuracy:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, ROC AUC = {roc:.4f}")

    # 混淆矩阵
    ConfusionMatrixDisplay.from_estimator(model, X, y, cmap=plt.cm.Blues, normalize='all')
    plt.title(f"{model_name} - Confusion Matrix")
    plt.show()

    # ROC曲线
    RocCurveDisplay.from_estimator(model, X, y)
    plt.title(f"{model_name} - ROC Curve")
    plt.show()

    return accuracy, precision, recall, roc

# ========== 1. 逻辑回归 ==========
params = {'C': [0.1, 0.5, 1, 5, 10]}
rscv = RandomizedSearchCV(estimator=LogisticRegression(max_iter=500), param_distributions=params, scoring=f1_scorer, n_iter=5, verbose=1)
rscv.fit(X, y)
params = rscv.best_params_
print("Best parameters for Logistic Regression:", params)
model_lr = LogisticRegression(max_iter=500, **params)
acc_lr, pre_lr, rec_lr, roc_lr = get_model_metrics_and_plot(model_lr, X, y, "Logistic Regression")

# ========== 2. 朴素贝叶斯 ==========
model_nb = GaussianNB()
acc_nb, pre_nb, rec_nb, roc_nb = get_model_metrics_and_plot(model_nb, X, y, "Naive Bayes")

# ========== 3. XGBoost ==========
params_xgb = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 150],
    'subsample': [0.6, 0.8, 1.0]
}
rscv_xgb = RandomizedSearchCV(
    estimator=XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    param_distributions=params_xgb,
    scoring=f1_scorer,
    n_iter=10,
    verbose=1,
    n_jobs=1
)
rscv_xgb.fit(X, y)
params_xgb = rscv_xgb.best_params_
print("Best parameters for XGBoost:", params_xgb)
model_xgb = XGBClassifier(**params_xgb, use_label_encoder=False, eval_metric='logloss')
acc_xgb, pre_xgb, rec_xgb, roc_xgb = get_model_metrics_and_plot(model_xgb, X, y, "XGBoost")

# ========== 4. LightGBM ==========
params_lgbm = {
    'num_leaves': [31, 50, 70],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 150]
}
rscv_lgbm = RandomizedSearchCV(
    estimator=LGBMClassifier(),
    param_distributions=params_lgbm,
    scoring=f1_scorer,
    n_iter=10,
    verbose=1,
    n_jobs=1
)
rscv_lgbm.fit(X, y)
params_lgbm = rscv_lgbm.best_params_
print("Best parameters for LightGBM:", params_lgbm)
model_lgbm = LGBMClassifier(**params_lgbm)
acc_lgbm, pre_lgbm, rec_lgbm, roc_lgbm = get_model_metrics_and_plot(model_lgbm, X, y, "LightGBM")

# ========== 交叉验证对比柱状图 ==========
model_names = ['Logistic Regression', 'Naive Bayes', 'XGBoost', 'LightGBM']
model_list = [model_lr, model_nb, model_xgb, model_lgbm]

cv_accuracies = []
cv_precisions = []
cv_recalls = []
cv_rocs = []
cv_accuracies_std = []
cv_precisions_std = []
cv_recalls_std = []
cv_rocs_std = []

print("\n===== 各模型5折交叉验证均值和标准差（百分数，保留两位小数）=====")
for name, model in zip(model_names, model_list):
    acc = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    pre = cross_val_score(model, X, y, cv=5, scoring='precision')
    rec = cross_val_score(model, X, y, cv=5, scoring='recall')
    roc = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    cv_accuracies.append(acc.mean())
    cv_precisions.append(pre.mean())
    cv_recalls.append(rec.mean())
    cv_rocs.append(roc.mean())
    cv_accuracies_std.append(acc.std())
    cv_precisions_std.append(pre.std())
    cv_recalls_std.append(rec.std())
    cv_rocs_std.append(roc.std())
    print(f"{name}: "
          f"Accuracy={acc.mean()*100:.2f}%±{acc.std()*100:.2f}%, "
          f"Precision={pre.mean()*100:.2f}%±{pre.std()*100:.2f}%, "
          f"Recall={rec.mean()*100:.2f}%±{rec.std()*100:.2f}%, "
          f"ROC AUC={roc.mean()*100:.2f}%±{roc.std()*100:.2f}%")

# ================== 可视化部分（百分数，保留两位小数） ==================
x = np.arange(len(model_names))  # 模型的索引
width = 0.2  # 柱状图的宽度

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - 1.5 * width, [v*100 for v in cv_accuracies], width, yerr=[s*100 for s in cv_accuracies_std], capsize=5, label='Accuracy')
rects2 = ax.bar(x - 0.5 * width, [v*100 for v in cv_precisions], width, yerr=[s*100 for s in cv_precisions_std], capsize=5, label='Precision')
rects3 = ax.bar(x + 0.5 * width, [v*100 for v in cv_recalls], width, yerr=[s*100 for s in cv_recalls_std], capsize=5, label='Recall')
rects4 = ax.bar(x + 1.5 * width, [v*100 for v in cv_rocs], width, yerr=[s*100 for s in cv_rocs_std], capsize=5, label='ROC AUC')

ax.set_xlabel('Models')
ax.set_ylabel('Scores (%)')
ax.set_title('Comparison of Model Performance (5-fold CV)')
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.legend()

def add_labels(rects, values):
    for rect, v in zip(rects, values):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., height,
                f'{v*100:.2f}%', ha='center', va='bottom')

add_labels(rects1, cv_accuracies)
add_labels(rects2, cv_precisions)
add_labels(rects3, cv_recalls)
add_labels(rects4, cv_rocs)

plt.tight_layout()
plt.show()

# 选择最优模型（以ROC AUC为准）
best_model_index = np.argmax(cv_rocs)
print(f"Best Model: {model_names[best_model_index]}")
print("ROC AUC 均值列表（百分数）:", [f"{v*100:.2f}%" for v in cv_rocs])
