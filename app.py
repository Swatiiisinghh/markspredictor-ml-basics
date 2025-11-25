from flask import Flask, request, jsonify, render_template, send_from_directory
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# ---------- LOAD MODEL + ENCODERS ----------
MODEL_PATH = os.path.join("models", "marks_model.pkl")
ENCODERS_PATH = os.path.join("models", "encoders.pkl")

model = joblib.load(MODEL_PATH)          # trained LinearRegression
encoders = joblib.load(ENCODERS_PATH)    # dict[col_name] = LabelEncoder

# Load CSV once to know column order + dtypes
df = pd.read_csv(os.path.join("data", "marks.csv"))
TARGET = "Exam_Score"

# All input feature columns (in the same order as training)
feature_columns = [col for col in df.columns if col != TARGET]

# Categorical vs numeric columns inferred from CSV
categorical_cols = [col for col in feature_columns if df[col].dtype == "object"]
numeric_cols = [col for col in feature_columns if col not in categorical_cols]


@app.route("/")
def home():
    # Serve React frontend
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Expects JSON with ALL 19 input fields, e.g.:

    {
      "Hours_Studied": 23,
      "Attendance": 84,
      "Parental_Involvement": "Low",
      "Access_to_Resources": "High",
      "Extracurricular_Activities": "No",
      "Sleep_Hours": 7,
      "Previous_Scores": 73,
      "Motivation_Level": "Low",
      "Internet_Access": "Yes",
      "Tutoring_Sessions": 0,
      "Family_Income": "Low",
      "Teacher_Quality": "Medium",
      "School_Type": "Public",
      "Peer_Influence": "Positive",
      "Physical_Activity": 3,
      "Learning_Disabilities": "No",
      "Parental_Education_Level": "High School",
      "Distance_from_Home": "Near",
      "Gender": "Male"
    }
    """
    data = request.get_json() or {}

    try:
        # 1. Check all required fields are present
        missing = [col for col in feature_columns if col not in data]
        if missing:
            return jsonify({
                "error": f"Missing fields: {', '.join(missing)}"
            }), 400

        # 2. Build feature row in correct order
        row = []

        for col in feature_columns:
            raw_val = data[col]

            if col in categorical_cols:
                # Ensure it's a string category value
                raw_val = str(raw_val)

                if col not in encoders:
                    return jsonify({"error": f"No encoder found for column '{col}'"}), 500

                le = encoders[col]
                # If user sends a category never seen in training, this will raise
                try:
                    encoded_val = le.transform([raw_val])[0]
                except ValueError:
                    allowed = ", ".join(map(str, le.classes_))
                    return jsonify({
                        "error": f"Invalid value '{raw_val}' for '{col}'. Allowed: {allowed}"
                    }), 400

                row.append(encoded_val)
            else:
                # numeric feature
                try:
                    row.append(float(raw_val))
                except ValueError:
                    return jsonify({
                        "error": f"Column '{col}' must be numeric, got '{raw_val}'"
                    }), 400

        # 3. Predict
        X = np.array([row])   # shape (1, n_features)
        pred = model.predict(X)[0]

        return jsonify({"prediction": float(pred)})

    except Exception as e:
        # Any unexpected error
        return jsonify({"error": str(e)}), 500


# ---------- ROUTE TO SERVE CHART IMAGES ----------
@app.route("/charts/<path:filename>")
def charts(filename):
    # Serve PNGs (histograms / scatter) from the models folder
    return send_from_directory("models", filename)


if __name__ == "__main__":
    print("✅ Loaded model from:", MODEL_PATH)
    print("✅ Loaded encoders from:", ENCODERS_PATH)
    print("✅ Feature columns:", feature_columns)
    app.run(debug=True)
