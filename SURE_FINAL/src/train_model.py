"""
train_model.py
--------------
1. Loads student_dataset.csv
2. Derives the Risk_Level target column via a composite scoring formula
3. Encodes all categorical features
4. Trains a Random Forest Classifier
5. Saves  →  risk_model.pkl | label_encoder.pkl | feature_columns.pkl | encoders.pkl
"""

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ── 1. Resolve paths & load data ──────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir    = os.path.abspath(os.path.join(current_dir, '..'))

data_candidates = [
    os.path.join(root_dir, 'data', 'student_dataset.csv'),
    os.path.join(current_dir, '..', 'data', 'student_dataset.csv'),
    os.path.join(current_dir, 'student_dataset.csv'),
    'student_dataset.csv'
]
data_path = next((p for p in data_candidates if os.path.exists(p)), 'student_dataset.csv')

df = pd.read_csv(data_path)
print(f"Loaded dataset from {data_path}: {df.shape[0]} rows x {df.shape[1]} columns")

# ── 2. Derive Risk_Level target ───────────────────────────────────────────────
# Risk Score starts from Average_Marks (already on 0-100 scale).
# Penalties are applied for behavioural / financial risk factors.
def compute_risk_score(row):
    score = row['Average_Marks']                   # base score (0–100)

    # Attendance penalty
    if row['Attendance_Percent'] < 60:
        score -= 20
    elif row['Attendance_Percent'] < 75:
        score -= 10

    # Arrears penalty
    if row['Previous_Arrears'] >= 3:
        score -= 20
    elif row['Previous_Arrears'] >= 1:
        score -= 10

    # Financial penalty
    if row['Fees_Status'] == 'Not Paid':
        score -= 15
    elif row['Fees_Status'] == 'Partially Paid':
        score -= 7

    # Internet access (minor)
    if row['Internet_Access'] == 'No':
        score -= 3

    return score

df['Risk_Score'] = df.apply(compute_risk_score, axis=1)

def assign_risk(score):
    if score < 45:
        return 'High Risk'
    elif score < 65:
        return 'Medium Risk'
    else:
        return 'Low Risk'

df['Risk_Level'] = df['Risk_Score'].apply(assign_risk)

print("\nRisk Level Distribution:")
print(df['Risk_Level'].value_counts())

# ── 3. Prepare features ───────────────────────────────────────────────────────
# Drop columns not used as ML features
FEATURES = [
    'Age', 'Gender', 'Department', 'Residence', 'Guardian',
    'Family_Income', 'CAT1_Marks', 'CAT2_Marks', 'Model_Exam_Marks',
    'Average_Marks', 'Attendance_Percent', 'Previous_Arrears',
    'Fees_Status', 'Internet_Access'
]

CATEGORICAL = ['Gender', 'Department', 'Residence', 'Guardian',
               'Family_Income', 'Fees_Status', 'Internet_Access']

X = df[FEATURES].copy()
y = df['Risk_Level']

# Encode each categorical column and save its encoder
encoders = {}
for col in CATEGORICAL:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

# ── 4. Train / Test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ── 5. Train model ────────────────────────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',   # handles any class imbalance
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ── 6. Evaluate ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=target_encoder.classes_))

# Feature importance
fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nTop Feature Importances:")
print(fi.head(10).round(4))

# ── 7. Save artifacts ─────────────────────────────────────────────────────────
models_dir = os.path.join(root_dir, 'models')
os.makedirs(models_dir, exist_ok=True)

joblib.dump(model,          os.path.join(models_dir, 'risk_model.pkl'))
joblib.dump(target_encoder, os.path.join(models_dir, 'label_encoder.pkl'))
joblib.dump(FEATURES,       os.path.join(models_dir, 'feature_columns.pkl'))
joblib.dump(encoders,       os.path.join(models_dir, 'encoders.pkl'))

print(f"\nAll artifacts saved successfully in '{models_dir}':")
print("  risk_model.pkl | label_encoder.pkl | feature_columns.pkl | encoders.pkl")
