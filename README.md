# House-Price-Prediction
Machine Learning project for predicting California house prices using regression models and HistGradientBoosting.
# California House Price Prediction 🏠

## 📌 Project Overview

This project focuses on predicting house prices using the **California Housing dataset** and Machine Learning regression techniques.

The objective is to build a predictive model that estimates the median house value based on demographic, housing, and geographical features.

The project was developed using **Python and Google Colab**.

---

## 🎯 Objectives

* Understand and analyze a real-world housing dataset
* Perform Exploratory Data Analysis (EDA)
* Handle missing values and categorical features
* Build a preprocessing pipeline
* Train a baseline regression model
* Compare multiple Machine Learning models using cross-validation
* Perform hyperparameter tuning using GridSearchCV
* Evaluate the final model on unseen test data
* Build a function for predicting the price of a new house

---

## 📊 Dataset

The project uses the **California Housing Prices** dataset.

The dataset contains **20,640 records and 10 columns**.

### Features

| Feature              | Description                     |
| -------------------- | ------------------------------- |
| `longitude`          | Geographic longitude            |
| `latitude`           | Geographic latitude             |
| `housing_median_age` | Median age of houses in a block |
| `total_rooms`        | Total number of rooms           |
| `total_bedrooms`     | Total number of bedrooms        |
| `population`         | Population in the block         |
| `households`         | Number of households            |
| `median_income`      | Median income of households     |
| `ocean_proximity`    | Location relative to the ocean  |

### Target Variable

`median_house_value`

The target represents the median house value for households in a block.

---

## 🔍 Exploratory Data Analysis

The following EDA techniques were performed:

* Dataset structure and data types
* Missing value analysis
* Duplicate value analysis
* Descriptive statistics
* Categorical feature analysis
* Target variable distribution
* Feature distributions
* Outlier analysis using boxplots
* Correlation heatmap
* Correlation analysis with the target variable

The dataset contains missing values only in the `total_bedrooms` column, with **207 missing values**.
The analysis showed that `median_income` had the strongest numerical correlation with `median_house_value`, with a correlation of approximately **0.688**.

---

## 🛠️ Data Preprocessing

A Scikit-learn preprocessing pipeline was created to avoid data leakage.

### Numerical Features

The numerical preprocessing included:

* Median imputation for missing values
* StandardScaler for feature scaling

### Categorical Features

The categorical preprocessing included:

* Most-frequent imputation
* One-hot encoding using `OneHotEncoder`

The dataset was split into:

* **80% training data:** 16,512 records
* **20% testing data:** 4,128 records

---

## 🤖 Models Compared

The following regression models were evaluated using **5-fold cross-validation**:

* Linear Regression
* Ridge Regression
* Lasso Regression
* Random Forest Regressor
* HistGradientBoosting Regressor

### Cross-Validation Results

| Model                |   CV RMSE |    CV MAE | CV R² |
| -------------------- | --------: | --------: | ----: |
| HistGradientBoosting | 48,097.73 | 32,290.34 | 0.827 |
| Random Forest        | 49,550.69 | 32,323.86 | 0.816 |
| Ridge                | 68,595.62 | 49,664.33 | 0.648 |
| Lasso                | 68,603.23 | 49,667.26 | 0.648 |
| Linear Regression    | 68,604.16 | 49,667.16 | 0.648 |

The cross-validation comparison was based primarily on RMSE.

---

## ⚙️ Hyperparameter Tuning

`HistGradientBoostingRegressor` was selected for further optimization.

GridSearchCV with 5-fold cross-validation was used to search across combinations of:

* Learning rate
* Maximum depth
* Maximum leaf nodes
* Minimum samples per leaf
* L2 regularization

### Best Parameters

```text
learning_rate      = 0.1
max_depth          = None
max_leaf_nodes     = 63
min_samples_leaf   = 20
l2_regularization  = 0.1
```

The tuned model achieved a CV RMSE of approximately **47,408.38**.

---

## 📈 Final Model Performance

The tuned HistGradientBoosting model was retrained using the complete training set and evaluated on the unseen test set.

### Test Performance

| Metric   |         Result |
| -------- | -------------: |
| RMSE     | **46,642.894** |
| MAE      | **30,758.134** |
| R² Score |      **0.834** |

The model achieved an R² score of approximately **0.834** on the test dataset.

### Training Performance

| Metric   |     Result |
| -------- | ---------: |
| RMSE     | 35,823.721 |
| MAE      | 24,372.243 |
| R² Score |      0.904 |

---

## 🔮 Prediction System

A prediction function was created to estimate the median house value for a new house based on:

* Longitude
* Latitude
* Housing median age
* Total rooms
* Total bedrooms
* Population
* Households
* Median income
* Ocean proximity

The notebook demonstrates an example prediction of approximately:

```text
$447,805.32
```

---

## 📌 Key Insights

* `median_income` was the strongest numerical predictor of house value.
* `total_bedrooms` contained missing values and was handled through preprocessing.
* The target variable showed a right-skewed distribution and a capped upper value.
* Several numerical features contained skewness and outliers.
* Housing and population-related variables showed multicollinearity.
* Tree-based boosting performed better than the linear models evaluated in this project.

---

## 🚀 Future Improvements

Possible improvements include:

* Creating ratio-ba


