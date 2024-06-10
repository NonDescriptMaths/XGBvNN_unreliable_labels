import os

if not os.getcwd().endswith("src"):
    os.chdir("src")

import pandas as pd
from utils import get_data, get_X_y
import sklearn.linear_model as skl
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from preprocess import preprocess_data
from sklearn.metrics import (
    confusion_matrix,
    cohen_kappa_score,
    precision_recall_curve,
    f1_score,
)

df = pd.read_csv("../data/archive/Base.csv")
df = preprocess_data(df, log_scaling=True, drop_highly_correlated_features=True)

X = df.drop(['fraud_bool'], axis=1)
y = df['fraud_bool']

standard_transformer = Pipeline(steps=[("standard", StandardScaler())])
minmax_transformer = Pipeline(steps=[("minmax", MinMaxScaler(feature_range=(-1, 1)))])

# cols to scale:
# standard: prev_address_months_count, current_address_months_count, days_since_request, intended_balcon_amount, zip_count_4w, velocity_6h, velocity_24h, velocity_4w, bank_branch_count_8w, date_of_birth_distinct_emails_4w, bank_months_count, proposed_credit_limit,
# minmax: income, name_email_similarity, customer_age, credit_risk_score, session_length_in_minutes, device_distinct_emails_8w, device_fraud_count, month

preprocessor = ColumnTransformer(
    remainder="passthrough",
    transformers=[
        (
            "std",
            standard_transformer,
            [
                "prev_address_months_count",
                "current_address_months_count",
                "days_since_request",
                "intended_balcon_amount",
                "zip_count_4w",
                "velocity_6h",
                "velocity_24h",
                "velocity_4w",
                "bank_branch_count_8w",
                "date_of_birth_distinct_emails_4w",
                "bank_months_count",
                "proposed_credit_limit",
            ],
        ),
        (
            "minmax",
            minmax_transformer,
            [
                "income",
                "name_email_similarity",
                "customer_age",
                "credit_risk_score",
                "session_length_in_minutes",
                "device_distinct_emails_8w",
                "device_fraud_count",
                "month",
            ],
        ),
    ],
)

X_trans = preprocessor.fit_transform(X)

model = skl.LogisticRegression(class_weight="balanced")
model.fit(X_trans, y)

y_pred = model.predict(X_trans)
y_prob = model.predict_proba(X_trans)[:, 1]

df = pd.DataFrame(
    {
        "y": y,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }
)

df.to_csv("./preprocess/weaklearner_weights.csv")

# metrics:

cm = confusion_matrix(y, y_pred)
print("Confusion Matrix:")
print(cm)

kappa = cohen_kappa_score(y, y_pred)
print("Cohen's Kappa:")
print(kappa)

f1 = f1_score(y, y_pred)
print("F1 Score:")
print(f1)

from sklearn.metrics import roc_curve
import numpy as np

def recall_at_5pct_fpr(y_true, y_scores):
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Find the threshold nearest to the target 5% FPR
    target_fpr = 0.05
    nearest = np.argmin(np.abs(fpr - target_fpr))
    
    # Get the TPR (Recall) at the nearest FPR
    recall_5fpr = tpr[nearest]
    actual_fpr = fpr[nearest]  # Actual FPR at this threshold

    print(f"Actual FPR: {actual_fpr*100:.2f}%") #Print the actual FPR used as this function finds the closest actual FPR to 5%, which might not be exactly 5%.
    return recall_5fpr
print("Recall at 5pct FPR:")
print(recall_at_5pct_fpr(y, y_prob))

# Compute precision, recall, and thresholds
precision, recall, thresholds = precision_recall_curve(y, y_prob)
# Plot the precision-recall curve
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()