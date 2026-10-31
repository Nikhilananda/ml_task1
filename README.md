# Loan Default Risk Prediction

## 📌 Project Overview

Banks face significant financial risk when they approve loans that borrowers later fail to repay. This project builds a machine learning classification model to predict whether a loan applicant is likely to experience serious delinquency (default) in the near future.

Because only a small fraction of borrowers actually default, this is a classic **imbalanced classification problem**. A naive model predicting "no default" for every applicant would yield high overall accuracy but provide zero business value. Therefore, this project emphasizes specialized sampling techniques, custom prediction thresholds, and robust evaluation metrics to accurately identify high-risk applications early in the process.

## 📊 Dataset

The dataset contains historical loan application records.

* **Target Variable:** `Loan_Status` (Mapped to 1 for Default/Risk, 0 for Safe)
* **Key Features:** Applicant Income, Co-applicant Income, Loan Amount, Credit History, Education, Dependents, and Property Area.

## ⚙️ Methodology

The project pipeline follows a structured 11-step approach:

1. **Data Cleaning:** Handled missing values using median imputation for numerical features and mode imputation for categorical features.
2. **Exploratory Data Analysis (EDA):** Visualized feature distributions and relationships with the target variable to uncover underlying patterns.
3. **Feature Engineering:** One-hot encoded categorical variables and transformed ordinal features (e.g., `Dependents`).
4. **Handling Data Imbalance:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) strictly to the training data to generate synthetic examples of the minority class without causing data leakage into the test set.
5. **Model Selection & Training:** Trained a `RandomForestClassifier` utilizing balanced class weights to penalize the misclassification of actual defaults.
6. **Threshold Optimization:** Shifted the classification probability threshold away from the default 0.5 to prioritize recall, ensuring more high-risk loans are flagged.

## 📈 Evaluation Metrics

Because accuracy is a misleading metric for imbalanced data, the model's performance is heavily evaluated using:

* **Confusion Matrix:** To track True Positives, False Positives, True Negatives, and False Negatives.
* **Precision, Recall, and F1-Score:** Emphasizing recall to minimize the number of risky loans that slip through.
* **ROC-AUC Score:** To measure the model's overall ability to distinguish between safe and default classes.
* **Feature Importance:** To extract business insights regarding which application features most strongly indicate default risk.



## 🛠️ Technologies Used

* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, Imbalanced-Learn, Matplotlib, Seaborn
* **Environment:** Jupyter Notebook / Google Colab

Would you like me to draft the README for the Task 2 Transfer Learning project next?
