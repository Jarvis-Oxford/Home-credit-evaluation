# Home-credit-evaluation
Machine learning project for evaluating credit violation probability based on Kaggle competition dataset

## Credit Scoring: Predicting Loan Default Risk
This project builds a machine learning model to predict the likelihood of a customer defaulting on their debt within the next 2 years, based on historical credit and financial data.

It is based on the popular Kaggle “Give Me Some Credit” competition.

## Project Structure
├── Credit score.ipynb        # Main notebook with all modeling steps
├── cs-training.csv           # Training data
├── README.md                 # Project description (this file)

## Objectives
Predict SeriousDlqin2yrs: Whether a customer will default in the next 2 years.

Evaluate multiple models using cross-validation and ROC-AUC metrics.

Perform feature engineering, outlier handling, and threshold optimization.


| Feature                              | Description                                           |
| ------------------------------------ | ----------------------------------------------------- |
| SeriousDlqin2yrs (target)            | 1 = default in next 2 years                           |
| RevolvingUtilizationOfUnsecuredLines | Credit usage rate                                     |
| age                                  | Age of the customer                                   |
| DebtRatio                            | Monthly debt payments divided by gross monthly income |
| MonthlyIncome                        | Monthly income                                        |
| NumberOfOpenCreditLinesAndLoans      | Total open credit lines and loans                     |
| NumberOfTimes90DaysLate              | Number of times 90+ days late on a payment            |
| ...                                  | (Additional credit history features)                  |


## Workflow
1. Data Preprocessing
Imputation of missing income values

Outlier detection via:

Percentile-based

MAD

STD

Ensemble voting

Feature scaling via StandardScaler

2. Dimensionality Reduction (Visualization)
PCA / LDA / t-SNE for latent structure visualization

3. Model Training
LogisticRegression

KNeighborsClassifier

RandomForestClassifier

AdaBoostClassifier

GradientBoostingClassifier

4. Evaluation
Cross-validation using roc_auc

Confusion matrix visualization

ROC curve + threshold tuning

Best ROC cutoff via Euclidean distance


## Sample Results
Best ROC-AUC: ~0.87 using Gradient Boosting

Threshold tuning improves precision-recall tradeoff

Dimensionality reduction shows latent class separability


