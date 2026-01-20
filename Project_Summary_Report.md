
# Sleep Disorder Classification Project - Summary Report

## 1. Project Overview
This project aimed to predict sleep disorders (Insomnia, Sleep Apnea, or None) based on various health and lifestyle factors. The dataset contained 374 samples and 13 initial features.

## 2. Methodology
- **Data Preprocessing:** Handled missing values (NaN set to 'None') and performed feature engineering.
- **Feature Engineering:**
    - Split 'Blood Pressure' into `Systolic_BP` and `Diastolic_BP`.
    - Created `Stress_Sleep_Risk` (Stress Level * (10 - Sleep Duration)).
    - Mapped BMI categories to numeric values (`BMI_Numeric`).
- **Modeling:** Built a robust **Pipeline** combining `StandardScaler`, `OneHotEncoder`, and classification models.
- **Comparison:** Compared Logistic Regression and Random Forest models with hyperparameter tuning via `GridSearchCV`.

## 3. Key Findings
- **Most Important Features:**
    1. **BMI (Numeric Mapping):** Strongly correlated with sleep apnea.
    2. **Systolic Blood Pressure:** A key physiological metric for sleep health.
    3. **Diastolic Blood Pressure:** Highly predictive when combined with Systolic BP.
- **Model Performance:**
    - Both models achieved a high test accuracy of **~94.67%**.
    - **Logistic Regression** was selected as the final candidate due to its better cross-validation performance and inherent simplicity for deployment.

## 4. Evaluation Metrics (Logistic Regression)
| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Insomnia | 0.93 | 0.87 | 0.90 |
| None | 0.93 | 1.00 | 0.97 |
| Sleep Apnea | 1.00 | 0.88 | 0.93 |

## 5. Conclusions & Next Steps
- **Success:** The pipeline approach ensures that preprocessing is handled automatically during inference, preventing data leakage and simplifying production use.
- **Limitation:** The dataset is relatively small (374 samples), and the class 'None' is over-represented. Collecting more data for 'Insomnia' and 'Sleep Apnea' could further improve generalizability.
- **Next Steps:**
    - **Deployment:** The trained pipeline `best_sleep_disorder_model.joblib` can be integrated into a mobile or web app.
    - **Refinement:** Experiment with more complex models (e.g., XGBoost) or additional lifestyle features if available.
