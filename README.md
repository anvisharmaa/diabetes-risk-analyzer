# 🩺 Diabetes Risk Prediction Dashboard

> A machine learning platform that predicts diabetes risk using classifiers built **from scratch** — no ML libraries — trained on ~98,000 real patient records from the CDC.

**🔗 Live demo:** https://diabetes-risk-calculate.streamlit.app

Built by **Guillermo Novillo, Jorge, and Anvi** as a final project for DSA at the University of Florida.

---

## 📌 Overview

Most ML projects call `sklearn.fit()` and call it a day. This one doesn't.

We implemented **Logistic Regression** and **Linear SVM** from the ground up using only NumPy — including gradient descent, sigmoid activation, hinge loss, and a custom train/test split — then deployed them inside an interactive Streamlit dashboard where users enter their own health data and get a real-time diabetes risk prediction.

The dashboard also benchmarks our custom models against 5 standard scikit-learn classifiers (kNN, RBF SVM, Naive Bayes, LDA, QDA) and visualizes feature importance.

---

## 📈 Results

Evaluated on a held-out 20% test set (~19,600 records):

| Model | Accuracy | AUC-ROC |
|---|---|---|
| Logistic Regression (from scratch) | ~83.7% | ~0.906 |
| Linear SVM (from scratch) | ~83.7% | ~0.906 |

The dataset is moderately balanced (~61% non-diabetic / ~39% diabetic), so accuracy is a meaningful metric here, and an AUC-ROC around 0.91 indicates strong separation between the two classes.

---

## ✨ Features

### 🔢 Diabetes Risk Calculator
Enter age, BMI, HbA1c level, blood glucose, and other health indicators to get an instant prediction from both custom models.

### 🏆 Model Performance Comparison
Side-by-side benchmarking of 7 classifiers ranked by MSE, Accuracy, and AUC-ROC with interactive Plotly charts.

### 📊 Feature Importance
Visual breakdown of which health factors most influence the Logistic Regression model's predictions — color-coded by risk impact.

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| ML Models (from scratch) | NumPy, custom Logistic Regression & Linear SVM |
| Comparison Models | scikit-learn (kNN, SVM, Naive Bayes, LDA, QDA) |
| Data Processing | Pandas, NumPy |
| Frontend / Dashboard | Streamlit, Plotly, Altair |
| Dataset | CDC Diabetes Health Indicators (~98,000 records) |
| Deployment | Streamlit Community Cloud |
| Language | Python 3.11 (local) / 3.14 (Cloud) |

---

## 🧠 How the Models Work

Both models add their intercept (bias) term internally, so training and inference handle it identically.

### Logistic Regression (from scratch)
- Sigmoid activation with gradient clipping (`np.clip`) to prevent overflow
- Batch gradient descent with an adaptive learning rate that backs off if the loss spikes
- Convergence based on a tolerance threshold (`tol=1e-8`), up to 10,000 iterations

### Linear SVM (from scratch)
- Hinge loss with L2 regularization (`lambda_param=0.0005`)
- Subgradient descent optimization
- Same convergence logic as Logistic Regression

Both models are serialized with `pickle` and loaded by the dashboard for instant predictions. The fitted `StandardScaler` is saved alongside each model so new inputs are scaled exactly as in training.

---

## 🚀 Run Locally

> Python 3.11 recommended.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Retrain the custom models from scratch
python train_models.py

# 3. (Optional) Precompute the model-comparison data
python other_models.py

# 4. Launch the dashboard
streamlit run app.py
```

The pre-trained model files (`logistic_regression_model.pkl`, `svm_model.pkl`) are committed to the repo, so steps 2–3 are optional — the app runs out of the box. The comparison data is generated on demand the first time you open the Model Comparison page.

---

## 📁 Project Structure

```
├── main.py                  # Core ML logic: data loading, custom models
├── train_models.py          # Trains & saves Logistic Regression + SVM as .pkl files
├── other_models.py          # Trains sklearn comparison models, saves precomputed results
├── app.py                   # Streamlit dashboard (3 pages)
├── Graph_Testing.py         # Visualization experiments (scratch)
├── data/
│   └── diabetes_dataset.csv # CDC dataset (~98k records)
├── requirements.txt
├── runtime.txt              # Python version hint for Streamlit Cloud
├── .streamlit/config.toml   # Light theme + headless server
├── logistic_regression_model.pkl
└── svm_model.pkl
```

*University of Florida — Data Structures & Algorithms Final Project, Fall 2025*
