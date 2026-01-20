
"""
# Sleep Disorder Classification Project - 3 Model Version
# Models: Logistic Regression, Random Forest, Decision Tree
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings

warnings.filterwarnings('ignore')

# 1. Setup & Data Loading
df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')

# 2. Feature Engineering
df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
df['Stress_Sleep_Risk'] = df['Stress Level'] * (10 - df['Sleep Duration'])
bmi_map = {'Normal': 0, 'Normal Weight': 0, 'Overweight': 1, 'Obese': 2}
df['BMI_Numeric'] = df['BMI Category'].map(bmi_map)

le_target = LabelEncoder()
y = le_target.fit_transform(df['Sleep Disorder'])

numeric_features = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 
                    'Stress Level', 'Heart Rate', 'Daily Steps', 'Systolic_BP', 
                    'Diastolic_BP', 'Stress_Sleep_Risk', 'BMI_Numeric']
categorical_features = ['Gender', 'Occupation']
X = df[numeric_features + categorical_features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Pipelines
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
])

lr_pipe = Pipeline([('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=200))])
rf_pipe = Pipeline([('preprocessor', preprocessor), ('classifier', RandomForestClassifier())])
dt_pipe = Pipeline([('preprocessor', preprocessor), ('classifier', DecisionTreeClassifier())])

# 4. Training (GridSearch)
print("Training 3 Models...")
lr_grid = GridSearchCV(lr_pipe, {'classifier__C': [0.1, 1, 10]}, cv=5).fit(X_train, y_train)
rf_grid = GridSearchCV(rf_pipe, {'classifier__n_estimators': [100, 200]}, cv=5).fit(X_train, y_train)
dt_grid = GridSearchCV(dt_pipe, {'classifier__max_depth': [5, 10, None]}, cv=5).fit(X_train, y_train)

# 5. Evaluation
best_model = lr_grid.best_estimator_ # Choosing LR as final 
y_pred = best_model.predict(X_test)

print("\nFinal Test Accuracy (Best Model - LR):", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le_target.classes_))

# Save
joblib.dump(best_model, 'best_sleep_disorder_model.joblib')
print("Model Saved.")
