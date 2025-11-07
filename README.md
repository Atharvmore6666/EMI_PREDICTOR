# EMI_PREDICTOR
This project successfully developed machine learning models to address the problem of loan eligibility prediction and maximum monthly EMI calculation for an Intelligent Financial Risk Assessment Platform.



# EMI Predictor Project

## Project Type

EDA/Regression/Classification

## Contribution

Individual

## Name

Atharv More

## Project Summary

This project aims to develop a cutting-edge Intelligent Financial Risk Assessment Platform powered by machine learning (ML) to revolutionize the loan eligibility process. The platform leverages a rich dataset encompassing demographic factors, income streams, detailed expense breakdowns, existing debt obligations, and key financial health metrics to build robust predictive models for EMI eligibility and maximum affordability calculation.

## Problem Statement

Traditional lending and credit risk assessment processes often rely on limited, static indicators, leading to high default risk and inefficient processing. This project addresses the lack of an efficient, data-driven, and holistic system to accurately predict a customer's ability to service a new loan EMI by integrating granular financial, behavioral, and demographic data, thereby minimizing credit risk while maximizing responsible loan volume.

## Dataset

The dataset used for this project is `emi_prediction_dataset.csv`. It contains information on potential borrowers, including demographic details, financial status, existing obligations, and loan request parameters, along with their EMI eligibility status and maximum monthly EMI affordability.

## Methodology

The project followed a standard machine learning workflow:

1.  **Data Loading and Understanding**: The dataset was loaded and initially explored to understand its structure, variables, and identify missing values and duplicates.
2.  **Data Wrangling**:
    *   Handled mixed data types in columns ('age', 'monthly_salary', 'bank_balance') by converting them to numeric.
    *   Imputed missing values: Median imputation for numerical columns ('monthly_rent', 'credit_score', 'emergency_fund', 'bank_balance', 'age', 'monthly_salary') and mode imputation for the categorical 'education' column.
    *   Treated outliers in numerical features (excluding counts and ratios) using the IQR method with capping and flooring.
3.  **Feature Engineering**: Created a new feature 'total_monthly_expenses' and 'EMI_to_Salary_Ratio' to capture relevant financial aspects.
4.  **Categorical Encoding**: Applied one-hot encoding to categorical features.
5.  **Data Scaling**: Scaled numerical features using StandardScaler.
6.  **Data Splitting**: Split the dataset into training and testing sets (80/20 ratio).
7.  **Handling Imbalanced Dataset**: Addressed the class imbalance in the 'emi_eligibility' target variable using the SMOTE oversampling technique on the training data.
8.  **ML Model Implementation & Evaluation**:
    *   Implemented and evaluated several classification models for 'emi_eligibility': RandomForestClassifier, DecisionTreeClassifier, and XGBoostClassifier.
    *   Performed hyperparameter tuning on the XGBoost Classifier using RandomizedSearchCV to optimize performance, particularly focusing on metrics for minority classes (Eligible and High_Risk).
    *   Implemented and evaluated an XGBoost Regressor model to predict 'max_monthly_emi'.
9.  **Model Saving**: Saved the best performing classification model (Tuned XGBoost Classifier) and the XGBoost Regressor model, along with the scaler and label encoder, for future deployment.

## Results

*   **Classification Model (EMI Eligibility)**: The tuned XGBoost Classifier demonstrated the best performance, showing significant improvements in precision and F1-score for the 'High_Risk' class and better overall balanced performance (higher macro average F1-score) compared to other models. This indicates its effectiveness in accurately predicting loan eligibility status, including identifying high-risk applicants.
*   **Regression Model (Maximum Monthly EMI)**: The XGBoost Regressor achieved excellent results in predicting 'max_monthly_emi', with a very high R-squared score and low MSE/RMSE, suggesting strong predictive accuracy for a borrower's maximum affordable EMI.

## Business Impact

The developed models have the potential for significant positive business impact:

*   **Automated Underwriting**: Enables faster and more consistent loan application processing.
*   **Reduced Credit Defaults**: Improved identification of high-risk applicants minimizes the likelihood of non-performing assets.
*   **Personalized Lending**: Accurate prediction of eligibility and affordability allows for tailored loan offers.
*   **Enhanced Operational Efficiency**: Streamlines the lending process, freeing up resources.
*   **Increased Customer Trust**: Transparent and data-driven decisions build confidence.

## How to Run the Project

1.  Clone the repository.
2.  Install the required libraries (e.g., pandas, numpy, scikit-learn, xgboost, imbalanced-learn, matplotlib, seaborn, plotly). A `requirements.txt` file will be provided in the repository.
3.  Place the `emi_prediction_dataset.csv` file in the appropriate directory (or update the data loading path in the notebook).
4.  Run the Jupyter Notebook (`.ipynb` file) to reproduce the analysis and model building process.
5.  The saved models (`tuned_xgboost_model.joblib`, `xgboost_regressor_model.joblib`), scaler (`scaler.joblib`), and label encoder (`label_encoder.joblib`) can be used for deployment.


