# -Black-Friday-Purchase-Prediction-
Black Friday Purchase Prediction — EDA + AutoML (scikit-learn pipeline, PyCaret-free, Python 3.10+)
# 🛍️ Black Friday Purchase Prediction (PyCaret-Free)

This project performs **Exploratory Data Analysis (EDA)** and builds **machine learning models** to predict purchase amounts in the [Black Friday dataset](https://www.kaggle.com/datasets/mehdidag/black-friday).  

Originally implemented with **PyCaret**, this version has been refactored to use **pure scikit-learn pipelines** so it works smoothly on **Python 3.10, 3.11, and 3.12**.

---

## 📂 Project Structure
├── black-friday-eda-sklearn.ipynb # Main notebook (EDA + ML models, no PyCaret)
├── train.csv # Dataset
├── artifacts/ # (Generated) Model artifacts and results
│ ├── model_comparison.csv
│ ├── best_model.joblib
│ └── summary.json
├── requirements.txt # Dependencies
└── README.md # Project documentation
---

## ⚙️ Features
- End-to-end **EDA** with visualizations.
- **Data preprocessing** with `ColumnTransformer`:
  - Missing value imputation
  - Scaling numeric features
  - One-hot encoding categorical features
- **Model Zoo** (PyCaret-style `compare_models` replacement):
  - Linear Regression, Ridge, Lasso, ElasticNet
  - Random Forest, ExtraTrees, Gradient Boosting
  - KNN, SVR
  - XGBoost / LightGBM / CatBoost *(if installed)*
- **5-fold Cross Validation** for RMSE comparison
- **Best model selection** + holdout evaluation
- **Artifacts saved** (comparison CSV, trained model `.joblib`, summary JSON)
- Compatible with **Python 3.10+** (no PyCaret dependency)

---

## 📊 Dataset
The dataset `train.csv` contains anonymized user and purchase information from a Black Friday sale.  
- Rows: ~550,000  
- Target: **`Purchase`** (numeric — amount spent)  
- Features: Demographics, product categories, etc.  

---

## 🚀 Getting Started

### 1️⃣ Clone Repository
```bash
git clone https://github.com/your-username/black-friday-eda-ml.git
cd black-friday-eda-ml
pip install -r requirements.txt
jupyter notebook black-friday-eda-sklearn.ipynb

📝 Results
Models are compared via cross-validation RMSE.
The best model is automatically selected and evaluated on a holdout test set.
Artifacts (model, metrics, summary) are stored in the artifacts/ folder.

📦 Requirements
See requirements.txt
:pandas, numpy, scikit-learn, joblib
Optional: xgboost, lightgbm, catboost

pandas, numpy, scikit-learn, joblib

Optional: xgboost, lightgbm, catboost
