import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc
)

import pickle
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb

df = pd.read_csv('loan_data.csv')
print("✅ Data loaded:", df.shape)
print()
print("Missing values BEFORE cleaning:")
print(df.isnull().sum())
print()

df['income'] = df['income'].fillna(df['income'].median())
df['credit_score'] = df['credit_score'].fillna(df['credit_score'].median())
df['employment_years'] = df['employment_years'].fillna(df['employment_years'].median())

print("Missing values AFTER cleaning:")
print(df.isnull().sum())
print()

# Feature 1 — Debt to Income Ratio
df['debt_to_income'] = df['loan_amount'] / df['income']

# Feature 2 — Loan to Credit Score Ratio
df['loan_to_credit'] = df['loan_amount'] / df['credit_score']

df['high_risk'] = (
    (df['credit_score'] < 550) &
    (df['num_existing_loans'] >= 3)
).astype(int)

print("New features created:")
print(df[['debt_to_income','loan_to_credit','high_risk']].head())
print()

# ── ENCODE LOAN PURPOSE
le = LabelEncoder()
df['loan_purpose'] = le.fit_transform(df['loan_purpose'])
with open('label_encoder.pkl','wb') as f:
    pickle.dump(le, f)
print("Loan purpose encoding:", dict(zip(le.classes_, le.transform(le.classes_))))
print()

# ── FEATURES AND TARGET
X = df.drop('default', axis=1)
y = df['default']
feature_names = X.columns.tolist()
print("Features:", feature_names)

with open('feature_names.pkl','wb') as f:
    pickle.dump(feature_names, f)

# ── TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")
print()

# ── TRAIN RANDOM FOREST
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc  = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy : {rf_acc:.4f}")

# TRAIN XGBOOST ─────────────────────────────────
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    # WHY learning_rate=0.1?
    # XGBoost learns step by step — like studying slowly
    # High learning rate = learns fast but misses details
    # 0.1 is the standard starting value
    use_label_encoder=False,
    eval_metric='logloss',
    # logloss = logarithmic loss
    # standard evaluation metric for binary classification
    random_state=42
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_acc  = accuracy_score(y_test, xgb_pred)
print(f"XGBoost Accuracy       : {xgb_acc:.4f}")
print()

if xgb_acc >= rf_acc:
    best_model = xgb_model
    best_pred  = xgb_pred
    best_name  = "XGBoost"
else:
    best_model = rf
    best_pred  = rf_pred
    best_name  = "Random Forest"

print(f"Best model: {best_name}")
print()

print("── Classification Report ───────────")
print(classification_report(y_test, best_pred,
      target_names=['Will Repay','Will Default'],
      zero_division=0))

cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Will Repay','Will Default'],
            yticklabels=['Will Repay','Will Default'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix — {best_name}')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()
print("✅ Confusion matrix saved")

y_prob = best_model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

roc_auc = auc(fpr, tpr)

print(f"AUC Score: {roc_auc:.4f}")

# Draw the ROC curve
plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, color='#e74c3c', linewidth=2,
         label=f'{best_name} (AUC = {roc_auc:.3f})')
plt.plot([0,1],[0,1], 'k--', linewidth=1,
         label='Random Guess (AUC = 0.5)')

# The dotted diagonal line = random guessing
# Your model curve should be above this line
# The higher above = the better
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Loan Default Prediction')
plt.legend()
plt.tight_layout()
plt.savefig('roc_curve.png')
plt.close()
print("✅ ROC curve saved")
print()

# ── STEP 13: FEATURE IMPORTANCE ──────────────────────────
importances = best_model.feature_importances_
sorted_idx  = np.argsort(importances)
plt.figure(figsize=(8,5))
plt.barh([feature_names[i] for i in sorted_idx],
          importances[sorted_idx], color='#e74c3c')
plt.xlabel('Importance Score')
plt.title('Which factors affect loan default most?')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()
print("✅ Feature importance saved")

with open('model.pkl','wb') as f:
    pickle.dump(best_model, f)
with open('model_name.pkl','wb') as f:
    pickle.dump(best_name, f)
print(f"✅ Model saved — {best_name}")