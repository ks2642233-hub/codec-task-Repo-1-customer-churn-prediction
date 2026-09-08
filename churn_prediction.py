import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, classification_report, confusion_matrix, roc_curve

# loading the dataset
df = pd.read_csv("Telco-Customer-Churn.csv")
print(df.shape)
print(df.head())

# TotalCharges column has some empty strings instead of numbers, fixing that
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"])

# customerID is just an id, not useful for prediction
df = df.drop("customerID", axis=1)

print("churn rate:", df["Churn"].value_counts(normalize=True))

# quick EDA plots
plt.figure(figsize=(6,4))
sns.countplot(x="Churn", data=df)
plt.title("Churn count")
plt.savefig("outputs/churn_count.png")
plt.close()

plt.figure(figsize=(8,5))
pd.crosstab(df["Contract"], df["Churn"], normalize="index").plot(kind="bar", stacked=True)
plt.title("Churn by contract type")
plt.tight_layout()
plt.savefig("outputs/churn_by_contract.png")
plt.close()

plt.figure(figsize=(8,5))
sns.histplot(data=df, x="tenure", hue="Churn", bins=30)
plt.title("Tenure vs churn")
plt.savefig("outputs/tenure_vs_churn.png")
plt.close()

plt.figure(figsize=(6,5))
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Monthly charges vs churn")
plt.savefig("outputs/monthlycharges_vs_churn.png")
plt.close()

# now encoding categorical columns
target = df["Churn"].map({"Yes": 1, "No": 0})
X = df.drop("Churn", axis=1)

cat_cols = X.select_dtypes(include="object").columns
for col in cat_cols:
    X[col] = LabelEncoder().fit_transform(X[col])

num_cols = X.select_dtypes(exclude="object").columns
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=42, stratify=target)

print("train size", X_train.shape)
print("test size", X_test.shape)

# training 3 models like the guidelines asked

log_model = LogisticRegression(max_iter=1000, class_weight="balanced")
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)
log_prob = log_model.predict_proba(X_test)[:,1]

rf_model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight="balanced")
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:,1]

xgb_model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, eval_metric="logloss")
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_prob = xgb_model.predict_proba(X_test)[:,1]

print("\n--- Logistic Regression ---")
print("accuracy:", accuracy_score(y_test, log_pred))
print("recall:", recall_score(y_test, log_pred))
print("roc auc:", roc_auc_score(y_test, log_prob))
print(classification_report(y_test, log_pred))

print("\n--- Random Forest ---")
print("accuracy:", accuracy_score(y_test, rf_pred))
print("recall:", recall_score(y_test, rf_pred))
print("roc auc:", roc_auc_score(y_test, rf_prob))
print(classification_report(y_test, rf_pred))

print("\n--- XGBoost ---")
print("accuracy:", accuracy_score(y_test, xgb_pred))
print("recall:", recall_score(y_test, xgb_pred))
print("roc auc:", roc_auc_score(y_test, xgb_prob))
print(classification_report(y_test, xgb_pred))

# roc curve comparison
plt.figure(figsize=(7,6))
for name, prob in [("Logistic Regression", log_prob), ("Random Forest", rf_prob), ("XGBoost", xgb_prob)]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    plt.plot(fpr, tpr, label=name + " (auc=%.2f)" % auc)
plt.plot([0,1],[0,1],"--", color="gray")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.title("ROC comparison")
plt.savefig("outputs/roc_comparison.png")
plt.close()

# random forest gave best auc so going with that as final model
cm = confusion_matrix(y_test, rf_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("outputs/confusion_matrix.png")
plt.close()

feat_imp = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)[:10]
plt.figure(figsize=(8,6))
feat_imp.plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Top 10 features")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png")
plt.close()

print("\ndone, check outputs folder")
