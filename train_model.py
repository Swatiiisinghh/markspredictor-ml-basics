import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
import joblib
import matplotlib
matplotlib.use("Agg")  # so it can save plots without opening a window
import matplotlib.pyplot as plt

# ---------- 1. LOAD DATA ----------

# marks.csv is your StudentPerformanceFactors.csv
df = pd.read_csv("data/marks.csv")

target = "Exam_Score"           # what we want to predict
X = df.drop(columns=[target])
y = df[target]

# ---------- 2. ENCODE CATEGORICAL FEATURES ----------

encoders = {}

for col in X.columns:
    if X[col].dtype == "object":  # column with strings
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

# ---------- 3. TRAIN / TEST SPLIT ----------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- 4. TRAIN MODEL ----------

model = LinearRegression()
model.fit(X_train, y_train)

# ---------- 5. EVALUATION METRICS ----------

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print(f"R² Score : {r2:.4f}")
print(f"MAE      : {mae:.4f}")
print(f"MSE      : {mse:.4f}")
print(f"RMSE     : {rmse:.4f}")

# ---------- 6. SAVE MODEL & ENCODERS ----------

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/marks_model.pkl")
joblib.dump(encoders, "models/encoders.pkl")
print("✔ Model saved at    models/marks_model.pkl")
print("✔ Encoders saved at models/encoders.pkl")

# ---------- 7. PLOTS (HISTOGRAMS / SCATTER) ----------

# Histogram of Exam_Score
plt.figure()
df[target].hist(bins=20)
plt.title("Exam Score Distribution")
plt.xlabel("Exam_Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("models/exam_score_hist.png")
plt.close()

# Optional: histogram of Hours_Studied if column exists
if "Hours_Studied" in df.columns:
    plt.figure()
    df["Hours_Studied"].hist(bins=20)
    plt.title("Hours Studied Distribution")
    plt.xlabel("Hours_Studied")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("models/hours_studied_hist.png")
    plt.close()

# Scatter: Actual vs Predicted
plt.figure()
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Exam_Score")
plt.ylabel("Predicted Exam_Score")
plt.title("Actual vs Predicted Exam_Score")
plt.tight_layout()
plt.savefig("models/actual_vs_predicted.png")
plt.close()

print("✔ Plots saved in models/:")
print("   - exam_score_hist.png")
print("   - hours_studied_hist.png (if Hours_Studied exists)")
print("   - actual_vs_predicted.png")
