# 📘 MarksPredictor — ML + Flask + React-style Frontend

Predict student exam scores using **19 study & environment factors** from a Kaggle dataset.

This project includes:

- 🧠 A trained **Linear Regression** model (scikit-learn)  
- 🧩 Preprocessing with **LabelEncoder** for categorical features  
- 🌐 A **Flask** backend with a `/api/predict` endpoint  
- 🎨 A single-page **React-style** frontend (HTML + CSS + JSX via Babel)  
- 📊 Dataset visualizations (histograms + scatter plot)

#SCREENSHOTS

---<img width="1379" height="903" alt="Screenshot 2025-11-26 023914" src="https://github.com/user-attachments/assets/6c0675e0-56d1-450c-acee-6d702b7f93f3" />

<img width="1653" height="643" alt="Screenshot 2025-11-26 023929" src="https://github.com/user-attachments/assets/e692c2a5-c15c-4ac9-a5d2-3c7652c40cd8" />

## 🗂 Project Structure

```text
markspredictor-ml-basics/
│
├── data/
│   └── marks.csv                     # Kaggle Student Performance Factors dataset
│
├── models/
│   ├── marks_model.pkl               # Trained Linear Regression model
│   ├── encoders.pkl                  # Saved LabelEncoders for categorical features
│   ├── exam_score_hist.png           # Exam_Score distribution
│   ├── hours_studied_hist.png        # Hours_Studied distribution
│   └── actual_vs_predicted.png       # Actual vs Predicted Exam_Score
│
├── templates/
│   └── index.html                    # Frontend (HTML + CSS + React via Babel)
│
├── app.py                            # Flask API server
├── train_model.py                    # Model training + chart generation
└── requirements.txt                  # Python dependencies
