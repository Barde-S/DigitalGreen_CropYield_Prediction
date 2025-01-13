# Project README: DigitalGreen CropYield Prediction

## Overview

This project involves data analysis and predictive modeling for crop-related data. The primary focus is on data preparation, feature engineering, exploratory data analysis (EDA), and building machine learning models to predict and optimize crop yields. This README provides a step-by-step explanation of the codebase and methodology.

---

## Prerequisites

### Libraries Used:
- **Data Manipulation and Visualization**:
  - `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`
- **Data Preparation**:
  - `scipy`, `statsmodels`, `sklearn`, `ptitprince`
- **Machine Learning Models**:
  - `scikit-learn`, `lightgbm`, `xgboost`, `catboost`
- **Feature Selection**:
  - `mlxtend`

---

## Project Workflow

### 1. **Data Loading**
- Dataset is read from a CSV file using `pandas`.
- Initial exploration includes checking data types, missing values, duplicates, and descriptive statistics.

### 2. **Data Cleaning and Preprocessing**
- **Handling Missing Values**:
  - Columns with >50% missing data are dropped.
  - Categorical missing values are replaced with 'None' or 'Unknown'.
  - Numerical missing values are replaced with mean or mode.
- **Feature Engineering**:
  - Derived features like `TotalCost`, `CostPerAcre`, and `TotalBasalUsed` were created.
  - Redundant columns were dropped to reduce dimensionality.
- **Normalization and Scaling**:
  - Numerical data was normalized and scaled for better model performance.

### 3. **Exploratory Data Analysis (EDA)**
- **Correlation Analysis**:
  - Heatmap visualization of numerical feature correlations.
- **Boxplots**:
  - Distribution of crop production costs and yields across districts and land preparation methods.
- **Descriptive Statistics**:
  - Summary of numerical and categorical variables.

### 4. **Feature Selection**
- **Backward Elimination**:
  - Iteratively removes features with the highest p-value until all are below a defined significance level.
- **Exhaustive Feature Selection**:
  - Uses `mlxtend` to identify the optimal feature set for regression models.

### 5. **Model Building**
- **Train-Test Split**:
  - Data is split into 80% training and 20% testing sets.
- **Algorithms Used**:
  - Regression Models: `RandomForestRegressor`, `GradientBoostingRegressor`, `ElasticNet`, `Lasso`, etc.
  - Classification Models: `LogisticRegression`, `DecisionTreeClassifier`, `CatBoostClassifier`, etc.
- **Metrics Evaluated**:
  - Regression: Mean Absolute Error (MAE), Mean Squared Error (MSE), R² Score

### 6. **Optimization**
- GridSearchCV and Cross-Validation were used for hyperparameter tuning to enhance model performance.

---

## Key Features

1. **Robust Preprocessing**:
   - Handles missing, inconsistent, and redundant data effectively.
2. **Visualizations**:
   - Clear and insightful visualizations for data exploration.
3. **Feature Engineering**:
   - Derives impactful features that improve predictive models.
4. **Machine Learning**:
   - Comprehensive use of algorithms for both regression and classification tasks.
5. **Model Selection**:
   - Feature selection techniques to identify significant predictors.

---

## Instructions for Use

1. **Setup**:
   - Install the required libraries using `pip install -r requirements.txt`.
   - Replace the dataset path in the script with the path to your CSV file.
2. **Run the Script**:
   - Execute the script sequentially in a Jupyter notebook or Python IDE.
3. **Modify Features**:
   - Customize feature engineering and selection methods as per the dataset.
4. **Evaluate Models**:
   - Use the provided metrics to compare model performance.

---

## Future Enhancements

- Incorporate additional algorithms like `AutoML` for automated model selection.
- Extend feature engineering with domain-specific insights.
- Deploy models as a web app for real-time predictions.

---

## Contact

For queries or contributions, please contact: [mallamsz74@gmail.com]
