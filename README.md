Adult Autism Spectrum Disorder (ASD) Prediction Using Machine Learning
Overview

This project investigates the use of machine learning (ML) to predict Autism Spectrum Disorder (ASD) in adults using questionnaire responses (AQ-10 items) and demographic variables.
The goal is to develop a lightweight, interpretable, and scalable screening model that can be embedded in mobile or web-based applications, enabling early detection and supporting equitable healthcare.

Methodology
1. Data Acquisition

Dataset: Autism Prediction Dataset (Shinde, 2022, Kaggle)

Records: 3,723 (2,800 train, 923 test)

Features: AQ-10 questionnaire responses, age, gender, ethnicity, family ASD history, country of residence, auxiliary flags

Target: Binary classification (ASD / Not ASD)

2. Exploratory Data Analysis (EDA)

Distribution plots for categorical & numeric features

Class imbalance check (ASD ≈ 1:5 ratio)

Correlation heatmaps for AQ-10 items

Outlier checks in age

Subgroup analysis (gender, ethnicity, family history)

3. Preprocessing
Step	Methodology	Details
Data Cleaning	Placeholder Handling	Replaced “?” with Unknown, standardized categorical text
Missing Values	Imputation	Categorical → new “Unknown”; Numerical → none missing
Encoding	Label & One-Hot	Binary → label-encoded; Multi-class → one-hot encoded
Scaling	Min-Max Scaling	Applied to age and result
Outliers	IQR Filtering	Checked age, applied log-transform where needed
Feature Selection	Recursive Feature Elimination (RFE)	Dropped weak predictors (e.g., country_of_residence high-cardinality)
Class Balancing	SMOTE + Tomek	Synthetic oversampling of ASD-positive + noise removal
PCA (optional)	Visualization	Used for high-dimensional visualizations, not final model
4. Modelling

Baseline Models: Logistic Regression, SVM

Tree Ensembles: Random Forest, XGBoost, LightGBM

Stacking Model: Meta-classifier combining LR + RF + XGBoost → best results

Hyperparameter Tuning: GridSearchCV / RandomizedSearchCV

5. Interpretability & Fairness

Feature importance: SHAP analysis → AQ-10 items (A5, A9) most predictive

Fairness checks: Evaluated subgroup performance (gender, ethnicity) → no large disparities

Transparency: SHAP + LIME explanations provided for interpretability

Results

Best Model: Stacking (LR + RF + XGB)

Performance (example):

Accuracy: ~82%

Recall (ASD class): ~77%

AUROC: ~0.85

Interpretability: AQ-10 responses dominate prediction (expected clinically).

Fairness: Balanced recall across gender and ethnicity groups.

Reproducibility
Requirements

Python 3.10+

Libraries:

pip install -r requirements.txt


(Include: pandas, numpy, scikit-learn, xgboost, lightgbm, shap, matplotlib, seaborn, imbalanced-learn)

Steps to Reproduce

Clone the repository and navigate inside:

git clone <repo_url>
cd asd-prediction


Place train.csv and test.csv in the project root.

Run the preprocessing pipeline:

python preprocessing/preprocess.py


Outputs cleaned + balanced dataset.

Train models:

python models/train_models.py


Saves trained models in /models.

Evaluate:

python models/evaluate.py


Outputs metrics + fairness checks in /results.

(Optional) Launch Jupyter notebooks in /notebooks for detailed EDA/visualization.

Reproducibility Notes

Random Seeds: All models trained with fixed random seeds for replicability.

Cross-validation: Stratified K-Folds to ensure balanced splits.

Resampling: SMOTE + Tomek applied only within training folds to prevent leakage.

Pipelines: End-to-end preprocessing & modelling encapsulated in scikit-learn Pipeline objects.

References

Shinde, S. (2022). Autism Prediction Dataset. Kaggle.

Thabtah, F. (2018). Machine Learning in Autism Screening.

Rajagopalan et al. (2024). Large-scale ML for ASD. PubMed.

WHO (2023). Autism Spectrum Disorders Fact Sheet.

Batsakis et al. (2022). ML Methods for Adult ASD Diagnosis.
