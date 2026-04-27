# 🏏 Cricket Match Prediction Using Machine Learning

## 📌 About The Project

This project predicts the outcome of IPL matches using Machine Learning models.
It uses match-related features such as teams and match conditions to analyze whether a team is likely to win.

The main goal of this project is to understand how machine learning performs on sports data and how feature selection impacts prediction accuracy.

---

## 📊 Dataset

* Source: IPL Dataset
* Type: Match-based simplified dataset
* Features:

  * Team1
  * Team2
  * Match conditions (derived features)
* Target:

  * Match Result (Win/Loss)

---

## ⚙️ Workflow

1. Data Cleaning & Preprocessing
2. Feature Engineering
3. Feature Scaling (StandardScaler)
4. Model Training using multiple algorithms
5. Evaluation using Confusion Matrix & ROC Curve

---

## 🤖 Models Implemented

| Model               | Accuracy | F1 Score |
| ------------------- | -------- | -------- |
| Logistic Regression | 0.48     | 0.19     |
| KNN                 | 0.50 ⭐   | 0.44     |
| SVM                 | 0.46     | 0.14     |
| Random Forest       | 0.42     | 0.26     |

---

## 📈 Results & Analysis

* KNN performed the best among all models in terms of balanced performance.
* ROC-AUC values were around **0.43 – 0.50**, indicating performance close to random guessing.
* The models struggled due to limited and weak features.
* Confusion matrix shows overlapping classification between classes.

---

## 📊 Visualizations

* Model Performance Comparison Graph
* Confusion Matrix (KNN)
* ROC Curve for all models
* Toss decision insights (Bat First vs Bowl First)

---

## 📊 Key Insight

* Teams choosing to **bowl first win ~54% matches**
* Teams batting first win ~46% matches

👉 This suggests chasing has a slight advantage in IPL matches.

---

## 🧠 Key Learnings

* Feature quality is more important than model complexity
* Simple datasets lead to weak predictions
* Sports prediction requires richer data (players, form, venue history)
* Machine learning alone cannot perform well without strong features

---

## 🛠️ Built With

* Python
* Pandas & NumPy
* Scikit-learn
* Matplotlib & Seaborn
* VS Code

---

## 🚀 Future Improvements

* Add player-level statistics
* Include team form & win history
* Use advanced models (XGBoost, Deep Learning)
* Build a real-time prediction web app

---

## 👨‍💻 Author

**Brijesh Kumar**
