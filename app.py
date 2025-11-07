import streamlit as st
import joblib
import pandas as pd
import numpy as np
import warnings
from typing import Dict, Any, Tuple

# Suppress warnings that might appear from Streamlit or Joblib
warnings.filterwarnings('ignore')

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

# --- Model Loading (Cached for performance) ---
@st.cache_resource
def load_components():
    """
    Loads the pre-trained scaler, label encoder, and both XGBoost models.
    Uses st.cache_resource to load large objects only once.
    """
    try:
        scaler = joblib.load(SCALER_PATH)
        le = joblib.load(LABEL_ENCODER_PATH)
        classifier_model = joblib.load(CLASSIFIER_MODEL_PATH)
        regressor_model = joblib.load(REGRESSOR_MODEL_PATH)
        return scaler, le, classifier_model, regressor_model
    except FileNotFoundError as e:
        st.error(f"Error loading required file: {e.filename}. Please ensure all four model files are uploaded.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during model loading: {e}")
        st.stop()
    return None, None, None, None # Should not be reached if st.stop() works

# --- Preprocessing Function ---
def preprocess_data(raw_data: Dict[str, Any], scaler) -> np.ndarray:
    """
    Converts raw input dictionary into a scaled numpy array for the models.
    """
    # 1. Create a Pandas DataFrame to ensure correct structure and order
    df = pd.DataFrame([raw_data], columns=FEATURE_NAMES)
    
    # 2. Scale the features
    # NOTE: This returns a 17-column array, as the scaler was trained on all 17 features.
    scaled_features = scaler.transform(df)
    return scaled_features

# --- Prediction Function for Classification (Eligibility) ---
def predict_eligibility(scaled_features: np.ndarray, model: Any, label_encoder: Any) -> str:
    """
    Uses the classifier model to predict the eligibility class and decodes the result.
    """
    prediction_encoded = model.predict(scaled_features)
    predicted_label = label_encoder.inverse_transform(prediction_encoded)
    return predicted_label[0]

# --- Prediction Function for Regression (Max EMI Amount) ---
def predict_max_emi(scaled_features: np.ndarray, model: Any) -> float:
    """
    Uses the regressor model to predict the continuous Max EMI amount.
    """
    prediction_emi = model.predict(scaled_features)
    return round(prediction_emi[0], 2)

# --- Streamlit Application Layout ---
def main():
    st.set_page_config(page_title="Dual ML Model Predictor", layout="wide")

    st.title("Loan Eligibility & Max EMI Predictor")
    st.markdown("Enter the applicant's financial details below to get a risk assessment and a maximum recommended EMI amount.")

    # Load components
    scaler, label_encoder, classifier_model, regressor_model = load_components()

    # --- Input Form ---
    with st.form("loan_prediction_form"):
        st.subheader("Applicant Financial Data")
        
        # Use columns for a better, wider layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age (Years)", min_value=18, max_value=100, value=35, step=1)
            monthly_salary = st.number_input("Monthly Salary ($)", min_value=0.0, value=10000.0, step=100.0)
            years_of_employment = st.number_input("Years of Employment", min_value=0.0, value=10.0, step=0.5)
            monthly_rent = st.number_input("Monthly Rent/Mortgage ($)", min_value=0.0, value=1500.0, step=50.0)
            school_fees = st.number_input("School Fees ($)", min_value=0.0, value=0.0, step=100.0)
            college_fees = st.number_input("College Fees ($)", min_value=0.0, value=0.0, step=100.0)

        with col2:
            travel_expenses = st.number_input("Travel Expenses ($)", min_value=0.0, value=200.0, step=10.0)
            groceries_utilities = st.number_input("Groceries/Utilities ($)", min_value=0.0, value=800.0, step=50.0)
            other_monthly_expenses = st.number_input("Other Expenses ($)", min_value=0.0, value=300.0, step=10.0)
            current_emi_amount = st.number_input("Current EMI Amount ($)", min_value=0.0, value=500.0, step=10.0)
            credit_score = st.number_input("Credit Score (FICO)", min_value=300, max_value=850, value=750, step=1)
            bank_balance = st.number_input("Bank Balance ($)", min_value=0.0, value=25000.0, step=100.0)

        with col3:
            emergency_fund = st.number_input("Emergency Fund ($)", min_value=0.0, value=10000.0, step=100.0)
            requested_amount = st.number_input("Loan Requested Amount ($)", min_value=0.0, value=50000.0, step=1000.0)
            # max_monthly_emi is often an input feature but also the regression target.
            # We'll use a placeholder value that will be overridden by the prediction,
            # but is required for the scaler's input structure.
            max_monthly_emi_placeholder = st.number_input(
                "Max Monthly EMI (Placeholder, not used for prediction)", 
                min_value=0.0, value=1500.0, step=100.0, disabled=True
            )
            
            # --- Calculated Features (Required for the model input structure) ---
            # Calculate total expenses for the final two features
            total_monthly_expenses = (monthly_rent + school_fees + college_fees + 
                                      travel_expenses + groceries_utilities + 
                                      other_monthly_expenses + current_emi_amount)
            
            emi_to_salary_ratio = current_emi_amount / monthly_salary if monthly_salary > 0 else 0.0

            st.markdown("---")
            st.metric("Calculated Total Monthly Expenses", f"${total_monthly_expenses:,.2f}")
            st.metric("Calculated EMI/Salary Ratio", f"{emi_to_salary_ratio:.2f}")

        # --- Submission Button ---
        st.markdown("---")
        submitted = st.form_submit_button("Get Prediction", type="primary")

    if submitted:
        # 1. Gather all inputs into the required dictionary format
        sample_input = {
            'age': age,
            'monthly_salary': monthly_salary,
            'years_of_employment': years_of_employment,
            'monthly_rent': monthly_rent,
            'school_fees': school_fees,
            'college_fees': college_fees,
            'travel_expenses': travel_expenses,
            'groceries_utilities': groceries_utilities,
            'other_monthly_expenses': other_monthly_expenses,
            'current_emi_amount': current_emi_amount,
            'credit_score': credit_score,
            'bank_balance': bank_balance,
            'emergency_fund': emergency_fund,
            'requested_amount': requested_amount,
            'max_monthly_emi': max_monthly_emi_placeholder, # Use placeholder value
            'EMI_to_Salary_Ratio': emi_to_salary_ratio,
            'total_monthly_expenses': total_monthly_expenses
        }

        # 2. Preprocess the input data (returns 17 features)
        scaled_data = preprocess_data(sample_input, scaler)
        
        # --- FIX: Drop the 'max_monthly_emi' column before prediction ---
        # The models were likely trained on 16 features (excluding the target 'max_monthly_emi'), 
        # but the scaler requires 17 features as input. We remove the target column (index 14) here.
        MAX_EMI_FEATURE_INDEX = FEATURE_NAMES.index('max_monthly_emi') 
        scaled_data_for_model = np.delete(scaled_data, MAX_EMI_FEATURE_INDEX, axis=1)

        # 3. Make both predictions using the 16-feature array
        with st.spinner('Calculating Eligibility and Max EMI...'):
            eligibility_prediction = predict_eligibility(scaled_data_for_model, classifier_model, label_encoder)
            max_emi_prediction = predict_max_emi(scaled_data_for_model, regressor_model)

        # 4. Output the result in a nice box
        st.subheader("Prediction Results")
        
        col_eligibility, col_max_emi = st.columns(2)

        # Format Eligibility Output
        if eligibility_prediction == 'Eligible':
            status_color = 'green'
        elif eligibility_prediction == 'High_Risk':
            status_color = 'orange'
        else: # Not_Eligible
            status_color = 'red'

        col_eligibility.markdown(
            f"""
            <div style="padding: 20px; border: 2px solid {status_color}; border-radius: 10px; text-align: center;">
                <p style="font-size: 16px; margin: 0;">Loan Eligibility Status:</p>
                <h2 style="color: {status_color}; margin: 5px 0 0 0;">{eligibility_prediction}</h2>
            </div>
            """, unsafe_allow_html=True
        )
        
        # Format Max EMI Output
        col_max_emi.markdown(
            f"""
            <div style="padding: 20px; border: 2px solid #007BFF; border-radius: 10px; text-align: center;">
                <p style="font-size: 16px; margin: 0;">Predicted Max Recommended EMI:</p>
                <h2 style="color: #007BFF; margin: 5px 0 0 0;">${max_emi_prediction:,.2f} / month</h2>
            </div>
            """, unsafe_allow_html=True
        )
        
        st.success("Analysis complete!")

if __name__ == "__main__":
    main()
