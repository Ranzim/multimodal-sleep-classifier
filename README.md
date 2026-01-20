# Multimodal Sleep Disorder Classifier

This repository contains a comprehensive data science pipeline for classifying sleep disorders (Insomnia, Sleep Apnea, or None) using health and lifestyle metrics. 

## 📊 Project Overview
The project leverages a dataset of 374 samples with features such as age, sleep duration, physical activity, stress levels, and blood pressure. The goal is to provide a robust, automated classification system that can be integrated into health monitoring applications.

## 🚀 Key Features
- **End-to-End Pipeline:** Integrated preprocessing, scaling, and encoding using Scikit-Learn pipelines to ensure zero data leakage.
- **Advanced Feature Engineering:**
  - Automated extraction of Systolic and Diastolic blood pressure.
  - Custom `Stress_Sleep_Risk` metric calculation.
  - BMI category mapping to numerical scales.
- **Model Comparison:** implementation of Logistic Regression, Random Forest, and Decision Tree models with hyperparameter tuning via `GridSearchCV`.
- **Top Performance:** Achieved ~94.7% accuracy on test data.

## 📁 Repository Structure
- `Sleep_Disorder_Classification_Pipeline.py`: Main Python script containing the full pipeline from data loading to model saving.
- `Sleep_Disorder_Classification_Pipeline.ipynb`: Interactive Jupyter Notebook with detailed analysis and visualizations.
- `Project_Summary_Report.md`: A detailed report on methodology, key findings, and performance metrics.
- `Sleep_health_and_lifestyle_dataset.csv`: The dataset used for training and testing.
- `best_sleep_disorder_model.joblib`: The serialized final model ready for deployment.

## 🛠️ Installation & Usage

### Prerequisites
- Python 3.8+
- Required libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Ranzim/multimodal-sleep-classifier.git
   cd multimodal-sleep-classifier
   ```
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn joblib
   ```

### Running the Classifier
To train the model and generate reports:
```bash
python Sleep_Disorder_Classification_Pipeline.py
```

## 📈 Performance Summary
The **Logistic Regression** model was selected for its balance of simplicity and high performance.

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Insomnia | 0.93 | 0.87 | 0.90 |
| None | 0.93 | 1.00 | 0.97 |
| Sleep Apnea | 1.00 | 0.88 | 0.93 |

**Overall Test Accuracy:** 94.67%

## 📝 License
This project is open-source and available under the MIT License.
