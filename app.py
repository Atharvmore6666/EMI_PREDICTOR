import joblib
import pandas as pd
import numpy as np
import os

# --- Configuration: Define the paths to the uploaded files ---
SCALER_PATH = 'scaler.joblib'
LABEL_ENCODER_PATH = 'label_encoder.joblib'
CLASSIFIER_MODEL_PATH = 'tuned_xgboost_model.joblib'
REGRESSOR_MODEL_PATH = 'xgboost_regressor_model.joblib'

# Expected features based on the scaler's training data.
FEATURE_NAMES = [
    'age', 'monthly_salary', 'years_of_employment', 'monthly_rent',
    'school_fees', 'college_fees', 'travel_expenses', 'groceries_utilities',
    'other_monthly_expenses', 'current_emi_amount', 'credit_score',
    'bank_balance', 'emergency_fund', 'requested_amount', 'max_monthly_emi',
    'EMI_to_Salary_Ratio', 'total_monthly_expenses'
]

# --- Loading the Pipeline Components ---
def load_components():
    """Loads the pre-trained scaler, label encoder, and both XGBoost models."""
    try:
        scaler = joblib.load(SCALER_PATH)
        le = joblib.load(LABEL_ENCODER_PATH)
        classifier_model = joblib.load(CLASSIFIER_MODEL_PATH)
        regressor_model = joblib.load(REGRESSOR_MODEL_PATH)
        
        print("✅ All model components loaded successfully (Scaler, Label Encoder, Classifier, Regressor).")
        return scaler, le, classifier_model, regressor_model
    except FileNotFoundError as e:
        print(f"❌ Error loading file: {e}")
        print("Please ensure all four files are available in the current directory.")
        return None, None, None, None
    except Exception as e:
        print(f"❌ An unexpected error occurred during loading: {e}")
        return None, None, None, None

# --- Preprocessing Function (Shared by both models) ---
def preprocess_data(raw_data: dict, scaler):
    """
    Converts raw input dictionary into a scaled numpy array for the models.
    """
    # 1. Create a Pandas DataFrame to ensure correct structure and order
    df = pd.DataFrame([raw_data], columns=FEATURE_NAMES)
    
    # 2. Scale the features
    scaled_features = scaler.transform(df)
    
    return scaled_features

# --- Prediction Function for Classification (Eligibility) ---
def predict_eligibility(scaled_features, model, label_encoder):
    """
    Uses the classifier model to predict the eligibility class and decodes the result.
    """
    prediction_encoded = model.predict(scaled_features)
    predicted_label = label_encoder.inverse_transform(prediction_encoded)
    
    return predicted_label[0]

# --- Prediction Function for Regression (Max EMI Amount) ---
def predict_max_emi(scaled_features, model):
    """
    Uses the regressor model to predict the continuous Max EMI amount.
    """
    # The regressor model returns a numerical array
    prediction_emi = model.predict(scaled_features)
    
    # Return the first (and only) prediction value, rounded to 2 decimal places
    return round(prediction_emi[0], 2)

# --- Main Execution Block ---
if __name__ == "__main__":
    # Load components
    scaler, label_encoder, classifier_model, regressor_model = load_components()

    if not all([scaler, label_encoder, classifier_model, regressor_model]):
        print("\nStopping script due to loading errors.")
    else:
        # Example Input Data (Simulated a candidate for loan/risk assessment)
        # MUST contain 17 features in the correct order for the DataFrame
        sample_input = {
            'age': 35,
            'monthly_salary': 10000.0,
            'years_of_employment': 10.0,
            'monthly_rent': 1500.0,
            'school_fees': 0.0,
            'college_fees': 0.0,
            'travel_expenses': 200.0,
            'groceries_utilities': 800.0,
            'other_monthly_expenses': 300.0,
            'current_emi_amount': 500.0,
            'credit_score': 750.0,
            'bank_balance': 25000.0,
            'emergency_fund': 10000.0,
            'requested_amount': 50000.0,
            'max_monthly_emi': 1500.0, # Note: This feature is likely dropped or used as a placeholder in real training
            'EMI_to_Salary_Ratio': 0.05,
            'total_monthly_expenses': 3300.0
        }
        
        # 1. Preprocess the input data (shared step)
        print("\nPreprocessing input data...")
        scaled_data = preprocess_data(sample_input, scaler)
        print(f"Input data scaled: {scaled_data.shape[1]} features ready for models.")

        # 2. Make both predictions
        print("Making eligibility prediction...")
        eligibility_prediction = predict_eligibility(scaled_data, classifier_model, label_encoder)
        
        print("Making Max EMI prediction...")
        max_emi_prediction = predict_max_emi(scaled_data, regressor_model)

        # 3. Output the result
        print("\n--- Dual Model Prediction Results ---")
        print(f"Input Data:\n{pd.Series(sample_input).to_string()}")
        print("\n-------------------------------------")
        print(f"1. Loan Eligibility: \033[1m{eligibility_prediction}\033[0m")
        print(f"2. Predicted Max EMI: \033[1m${max_emi_prediction:,.2f}\033[0m (per month)")
        print("-------------------------------------")
