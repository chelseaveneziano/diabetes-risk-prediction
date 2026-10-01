# Diabetes Risk Prediction with Machine Learning

Predicting whether a patient has diabetes from routine clinical measurements, and comparing three classification models to find the most reliable one for early screening.

## Overview

Many people live with diabetes without knowing it, and earlier detection leads to earlier care. This project builds a model that flags patients who may be at risk, using health metrics that are already collected in a standard checkup. Three models are compared: Logistic Regression, Random Forest, and XGBoost.

## Data

- **Source:** [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) (Kaggle)
- **Size:** 768 patient records, 8 clinical features
- **Features:** Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age
- **Target:** `Outcome` (1 = diagnosed with diabetes, 0 = not diagnosed)

About 35% of the patients in the dataset have a diabetes diagnosis.

## Approach

1. **Cleaning.** Zero values in Glucose, Blood Pressure, Skin Thickness, Insulin, and BMI are not realistic, so they were treated as missing and replaced with the median.
2. **Outliers.** Extreme values were capped at the 99th percentile.
3. **Exploration.** Histograms, a correlation heatmap, and a Glucose vs. BMI scatter plot were used to compare diabetic and non-diabetic patients.
4. **Preparation.** Features were standardized, and the data was split 80% training / 20% test.
5. **Modeling.** Logistic Regression served as the baseline, with a lowered decision threshold and balanced class weights tested to improve recall. Random Forest and XGBoost were then trained for comparison.
6. **Evaluation.** Models were compared on accuracy, precision, recall, F1-score, and ROC-AUC, with extra attention on recall for diabetic patients, since a missed case is the costlier error.

## Results

Scores for the diabetic class (Outcome = 1) on the 154-patient test set:

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 77% | 68% | 65% | 0.67 |
| Random Forest | 77% | 67% | 69% | 0.68 |
| XGBoost | 73% | 61% | 73% | 0.66 |

**Random Forest was chosen as the final model.** It tied for the highest accuracy and had the best balance of precision and recall. XGBoost caught slightly more diabetic patients, but it did so with more false positives and lower overall accuracy. Logistic Regression was accurate overall but missed more true diabetes cases.

### What drives the predictions

Random Forest feature importance scores:

| Feature | Importance |
|---|---|
| Glucose | 0.269 |
| BMI | 0.172 |
| Age | 0.145 |
| Diabetes Pedigree Function | 0.114 |
| Blood Pressure | 0.081 |
| Skin Thickness | 0.076 |
| Pregnancies | 0.072 |
| Insulin | 0.071 |

Glucose and BMI are the strongest predictors, which matches established clinical risk factors for diabetes.

## Limitations and Next Steps

- The dataset is small and covers only adult women of Pima Indian heritage, so the results may not generalize to other populations.
- Tune model hyperparameters to reduce misclassifications.
- Add features such as family history detail or physical activity.
- Test additional models, such as support vector machines or neural networks.

## How to Run

1. Download this repo. The dataset `diabetes.csv` is included.
2. Install the required libraries:
```
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```
3. Open `diabetes_prediction.ipynb` in Jupyter Notebook and run all cells.

## Files

- `diabetes_prediction.ipynb`: full analysis, from cleaning through model comparison
- `diabetes_prediction_presentation.pptx`: slide deck summarizing the project
- `diabetes.csv`: the dataset

## Tools

Python, pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Seaborn, Jupyter Notebook
