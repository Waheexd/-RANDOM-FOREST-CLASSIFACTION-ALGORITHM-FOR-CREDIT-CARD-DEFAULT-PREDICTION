💳 Credit Card Default Prediction

This project focuses on predicting credit card payment defaults using machine learning techniques.
It follows a structured workflow consisting of dataset preprocessing, feature construction, ensemble model optimization, explainability, and robustness evaluation — all implemented in Python using JupyterLab.

📘 Phase 1 – Dataset Selection, Preprocessing & Feature Construction
1. Dataset Selection

Dataset: Default of Credit Card Clients Dataset

Source: UCI Machine Learning Repository

Justification: The dataset contains over 30,000 records and 23 features, making it suitable for classification tasks and financial risk modeling.

2. Preprocessing & Feature Construction

Missing Values: Handled using mean/median imputation for numeric fields and mode/“unknown” for categorical values.

Outliers: Identified and capped using the IQR (Interquartile Range) method.

Encoding: Applied LabelEncoder and OneHotEncoder for categorical variables like gender, education, and marital status.

Derived Features: Created at least two new features — such as ratio-based or polynomial interaction terms — to capture meaningful relationships.

Multicollinearity Check: Performed using VIF and correlation heatmap; redundant variables were removed.

Normalization/Standardization: Used StandardScaler to bring features to comparable ranges.

🧠 Phase 2 – Ensemble Model Design, Optimization & Explainability
1. Model Design

Built an end-to-end Pipeline integrating preprocessing and model training.

Implemented RandomForestClassifier with nested cross-validation for hyperparameter tuning.

Parameters optimized:

n_estimators: {200, 400, 800}

max_depth: {10, 20, None}

max_features: {"sqrt", "log2"}

min_samples_split: {2, 4, 8}

class_weight: {"balanced", None}

Optimization Objectives: Macro-averaged F1-score and ROC-AUC.

Recorded results with best parameters, mean CV score, standard deviation, and runtime.

2. Model Explainability

Used Permutation Importance and/or SHAP for interpretability.

Global Explanation: Visualized key features influencing default predictions using feature importance or SHAP summary plots.

Local Explanation: Analyzed individual predictions using SHAP force or waterfall plots.

📊 Phase 3 – Evaluation, Robustness & Error Analysis
1. Evaluation Metrics

Computed on test data:

Accuracy

Precision

Recall

F1-score

ROC-AUC

Visualized using:

Confusion Matrix

ROC Curves for both training and testing data.

2. Robustness Check

Introduced 10% label noise by randomly flipping a subset of target labels.

Retrained the model and observed performance degradation to evaluate robustness.

Discussed issues related to overfitting, underfitting, and bias/fairness.

⚙️ Tools & Libraries

Language: Python

Environment: JupyterLab

Libraries Used:

NumPy

Pandas

Matplotlib

Seaborn

Scikit-learn

SHAP

📈 Results Summary

The optimized Random Forest model achieved strong predictive performance with balanced F1 and ROC-AUC scores.

Explainability analysis highlighted key factors such as payment history, credit limit, and bill amount as major drivers of default risk.

The model remained stable under label noise, demonstrating good robustness and reliability.
