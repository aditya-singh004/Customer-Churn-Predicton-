# Churn Prediction Model

This project builds a machine learning pipeline to predict customer churn (`Exited`) using structured banking customer data.

The full workflow is implemented in:
- `churn_prediction_using_machine_learning.ipynb`

## Project Overview

The notebook covers:
- Data loading and exploratory analysis
- Feature engineering and preprocessing
- Baseline model comparison with cross-validation
- Hyperparameter tuning
- Final model training and evaluation

The goal is to identify customers likely to churn so retention actions can be taken early.

## Dataset

- **File:** `churn (2).csv`
- **Target column:** `Exited` (`1` = churn, `0` = not churn)
- **Examples of input features:** `CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`, `Geography`, `Gender`

## Modeling Approach

### 1. Preprocessing
- Drops non-predictive identity columns such as `CustomerId` and `Surname`
- Applies robust scaling to selected numerical columns
- Uses train/test split with stratification (`test_size=0.20`, `random_state=42`)

### 2. Baseline Models (10-fold ROC-AUC)
- Logistic Regression: `0.7860`
- KNN: `0.7186`
- Decision Tree (CART): `0.6804`
- Random Forest: `0.8498`
- SVC: `0.7936`
- Gradient Boosting: `0.8656`
- LightGBM: `0.8586`

### 3. Hyperparameter Tuning
- **LightGBM (GridSearchCV)** best params:
  - `colsample_bytree=0.6`
  - `learning_rate=0.05`
  - `max_depth=5`
  - `n_estimators=100`
- **Gradient Boosting (GridSearchCV)** best params:
  - `learning_rate=0.1`
  - `max_depth=3`
  - `n_estimators=100`
  - `subsample=1.0`

### 4. Tuned Model Comparison (10-fold ROC-AUC)
- LightGBM: `0.868449`
- Gradient Boosting: `0.865587`

## Final Model Performance

Final model: **LightGBM** (best estimator from tuning)

Test set metrics:
- Accuracy: `0.87`
- ROC-AUC: `0.8691`
- Class `1` (churn): Precision `0.79`, Recall `0.47`, F1-score `0.59`

## Tech Stack

- Python
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost
- LightGBM

## How to Run

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm jupyter
```

3. Launch Jupyter:

```bash
jupyter notebook
```

4. Open and run:
- `churn_prediction_using_machine_learning.ipynb`

## Notes

- The notebook appears to have been developed in Google Colab and then exported, so some Colab-specific output cells may be present.
- If your local file name differs, update the dataset path in the notebook before running.
