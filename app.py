import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set the page configuration for a wider layout
st.set_page_config(layout="wide")

# --- Model Loading ---
try:
    # Load the trained models, scaler, and label encoder
    tuned_xgb_model = joblib.load('tuned_xgboost_model.joblib')
    xgb_reg_model = joblib.load('xgboost_regressor_model.joblib')
    scaler = joblib.load('scaler.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
    st.sidebar.success("✅ Models and preprocessing objects loaded!")
except FileNotFoundError:
    st.error("❌ Error: Model or preprocessing files not found. Please ensure 'tuned_xgboost_model.joblib', 'xgboost_regressor_model.joblib', 'scaler.joblib', and 'label_encoder.joblib' are in the same directory.")
    st.stop()


# --- Feature Definition (Kept as per original logic) ---
# IMPORTANT: These lists must be consistent with the features the loaded models were trained on.
important_features = [
    'requested_amount', 'max_monthly_emi', 'requested_tenure', 'EMI_to_Salary_Ratio',
    'monthly_salary', 'groceries_utilities', 'bank_balance', 'existing_loans_Yes',
    'total_monthly_expenses', 'current_emi_amount', 'travel_expenses', 'emergency_fund',
    'credit_score', 'other_monthly_expenses', 'years_of_employment', 'monthly_rent',
    'school_fees', 'college_fees', 'age',
    # Include relevant one-hot encoded columns if they were important
    'gender_Male', 'marital_status_Single', 'education_Professional',
    'employment_type_Private', 'company_type_MNC', 'house_type_Rented',
    'emi_scenario_Personal Loan EMI', 'emi_scenario_Vehicle EMI'
]

# Assuming a placeholder for categorical columns that were dropped first, 
# which is needed for proper one-hot encoding in the app logic.
# In a real scenario, these must be defined based on the training set.
categorical_cols_app = ['gender', 'marital_status', 'education', 'employment_type', 'company_type', 'house_type', 'existing_loans', 'emi_scenario'] 


regression_features_app = [f for f in important_features if f != 'max_monthly_emi']
classification_features_app = important_features

input_fields_to_show = [
    'age', 'monthly_salary', 'years_of_employment', 'monthly_rent',
    'family_size', 'dependents', 'school_fees', 'college_fees',
    'travel_expenses', 'groceries_utilities', 'other_monthly_expenses',
    'current_emi_amount', 'credit_score', 'bank_balance', 'emergency_fund',
    'requested_amount', 'requested_tenure',
    'gender', 'marital_status', 'education', 'employment_type', 'company_type',
    'house_type', 'existing_loans', 'emi_scenario'
]


# --- Streamlit UI: Title and Description ---
st.title("🏦 Intelligent Financial Risk Assessment Platform")
st.markdown("""
<style>
.stButton>button {
    font-size: 1.2em;
    font-weight: bold;
    color: white;
    background-color: #4CAF50;
    border-radius: 12px;
    padding: 10px 24px;
}
.stMetric {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
}
</style>
""", unsafe_allow_html=True)
st.markdown("---")

st.info("Enter the applicant's details below. The platform will predict the maximum monthly EMI they can afford and their overall EMI eligibility based on the requested loan.")

# --- Streamlit UI: Input Fields ---
input_data = {}

# Use st.container for better visual grouping
with st.container():
    st.subheader("👤 Personal & Financial Details")
    col1, col2, col3 = st.columns(3)

    # Column 1: Core Financials
    with col1:
        input_data['monthly_salary'] = st.number_input("💰 Monthly Salary (₹)", min_value=0.0, value=50000.0, step=1000.0)
        input_data['requested_amount'] = st.number_input("💵 Requested Loan Amount (₹)", min_value=0.0, value=500000.0, step=10000.0)
        input_data['requested_tenure'] = st.number_input("⏳ Requested Tenure (Months)", min_value=1, value=36, step=6)
        input_data['current_emi_amount'] = st.number_input("💸 Current Total EMI Amount (₹)", min_value=0.0, value=0.0, step=100.0)

    # Column 2: Expenses & Credit
    with col2:
        input_data['monthly_rent'] = st.number_input("🏠 Monthly Rent/Mortgage (₹)", min_value=0.0, value=10000.0, step=500.0)
        input_data['groceries_utilities'] = st.number_input("🛒 Groceries & Utilities (₹)", min_value=0.0, value=15000.0, step=500.0)
        input_data['travel_expenses'] = st.number_input("✈️ Travel Expenses (₹)", min_value=0.0, value=5000.0, step=100.0)
        input_data['other_monthly_expenses'] = st.number_input("🧾 Other Monthly Expenses (₹)", min_value=0.0, value=8000.0, step=100.0)
    
    # Column 3: Savings & Demographic
    with col3:
        input_data['credit_score'] = st.number_input("📊 Credit Score", min_value=300, max_value=900, value=700, step=1)
        input_data['bank_balance'] = st.number_input("🏦 Bank Balance (₹)", min_value=0.0, value=100000.0, step=1000.0)
        input_data['emergency_fund'] = st.number_input("🚨 Emergency Fund (₹)", min_value=0.0, value=50000.0, step=1000.0)
        input_data['years_of_employment'] = st.number_input("💼 Years of Employment", min_value=0.0, value=5.0, step=0.5)

st.markdown("---")

with st.container():
    st.subheader("👨‍👩‍👧‍👦 Applicant & Loan Context")
    col4, col5, col6 = st.columns(3)
    
    # Column 4: Personal Info
    with col4:
        input_data['age'] = st.number_input('Age (Years)', min_value=18, max_value=100, value=30, step=1)
        input_data['gender'] = st.selectbox('Gender', ['Male', 'Female'])
        input_data['marital_status'] = st.selectbox('Marital Status', ['Single', 'Married'])
        input_data['existing_loans'] = st.selectbox('Has Existing Loans?', ['Yes', 'No'])

    # Column 5: Family & Education
    with col5:
        input_data['family_size'] = st.number_input('Family Size', min_value=1, value=3, step=1)
        input_data['dependents'] = st.number_input('Number of Dependents', min_value=0, value=2, step=1)
        input_data['education'] = st.selectbox('Education', ['Professional', 'Graduate', 'Post Graduate', 'High School'])
        input_data['school_fees'] = st.number_input('🎓 School Fees (₹)', min_value=0.0, value=0.0, step=100.0)

    # Column 6: Employment & Housing
    with col6:
        input_data['employment_type'] = st.selectbox('Employment Type', ['Private', 'Government', 'Self-employed'])
        input_data['company_type'] = st.selectbox('Company Type', ['MNC', 'Mid-size', 'Startup', 'Large Indian', 'Small'])
        input_data['house_type'] = st.selectbox('House Type', ['Rented', 'Family', 'Own'])
        input_data['emi_scenario'] = st.selectbox('Loan Purpose/Scenario', ['Personal Loan EMI', 'Vehicle EMI', 'Education EMI', 'E-commerce Shopping EMI', 'Home Appliances EMI'])
        input_data['college_fees'] = st.number_input('🏫 College Fees (₹)', min_value=0.0, value=0.0, step=100.0) # Moved here for layout balance


st.markdown("---")
# --- Prediction Button ---
if st.button("🚀 Analyze Financial Risk and Eligibility"):
    
    # --- Data Preparation (Preserving Original Logic) ---
    with st.spinner('Analyzing applicant data and running models...'):
        input_df_original = pd.DataFrame([input_data])
        
        # --- Preprocessing for Regression Model (Max EMI Prediction) ---
        input_df_reg_processed = input_df_original.copy()

        # Handle categorical features
        categorical_cols_reg_app = [col for col in categorical_cols_app if col in input_df_reg_processed.columns]
        for col in categorical_cols_reg_app:
            # Standardize gender for OHE consistency, if not done in training (assuming M/F mapping)
            if col == 'gender':
                 input_df_reg_processed[col] = input_df_reg_processed[col].astype(str).str.upper().str[0]
                 input_df_reg_processed[col] = input_df_reg_processed[col].replace({'F': 'Female', 'M': 'Male'})
            input_df_reg_processed[col] = input_df_reg_processed[col].astype('category')
        input_df_reg_processed = pd.get_dummies(input_df_reg_processed, columns=categorical_cols_reg_app, drop_first=True)

        # Calculate engineered features (total_monthly_expenses is needed for regression)
        input_df_reg_processed['total_monthly_expenses'] = input_data['monthly_rent'] + input_data['school_fees'] + input_data['college_fees'] + input_data['travel_expenses'] + input_data['groceries_utilities'] + input_data['other_monthly_expenses']

        # Ensure columns match training data for regression
        missing_cols_reg = set(regression_features_app) - set(input_df_reg_processed.columns)
        for c in missing_cols_reg:
            input_df_reg_processed[c] = 0
        input_df_reg_processed = input_df_reg_processed[regression_features_app]

        # Apply the loaded scaler to the appropriate numerical columns for regression
        numerical_cols_for_scaling_reg = [col for col in regression_features_app if col in ['age', 'monthly_salary', 'years_of_employment', 'monthly_rent',
                                                                                            'school_fees', 'college_fees', 'travel_expenses', 'groceries_utilities',
                                                                                            'other_monthly_expenses', 'current_emi_amount', 'credit_score', 'bank_balance',
                                                                                            'emergency_fund', 'requested_amount', 'total_monthly_expenses']]

        cols_to_scale_reg_exist = [col for col in numerical_cols_for_scaling_reg if col in input_df_reg_processed.columns]

        if cols_to_scale_reg_exist:
            # Use the scaler on the *entire* input dataframe and then select the scaled columns
            # This is a robust way to handle the scaler trained on a larger feature set.
            # However, for simplicity and adherence to the original code's explicit column selection:
            input_df_reg_processed_scaled_part = scaler.transform(input_df_reg_processed[cols_to_scale_reg_exist])
            input_df_reg_processed[cols_to_scale_reg_exist] = input_df_reg_processed_scaled_part
        
        # --- Predict Max Monthly EMI (Regression Model) ---
        predicted_max_monthly_emi_scaled = xgb_reg_model.predict(input_df_reg_processed)
        
        # TEMPORARY SIMPLIFICATION: Assume regression model output is already unscaled for app demo (as per original logic)
        predicted_max_monthly_emi = predicted_max_monthly_emi_scaled[0]
        

        # --- Preprocessing for Classification Model (Eligibility Prediction) ---
        input_df_clf_processed = input_df_original.copy()

        # Handle categorical features (same as regression preprocessing)
        categorical_cols_clf_app = [col for col in categorical_cols_app if col in input_df_clf_processed.columns]
        for col in categorical_cols_clf_app:
            if col == 'gender':
                 input_df_clf_processed[col] = input_df_clf_processed[col].astype(str).str.upper().str[0]
                 input_df_clf_processed[col] = input_df_clf_processed[col].replace({'F': 'Female', 'M': 'Male'})
            input_df_clf_processed[col] = input_df_clf_processed[col].astype('category')
        input_df_clf_processed = pd.get_dummies(input_df_clf_processed, columns=categorical_cols_clf_app, drop_first=True)

        # Add the predicted max_monthly_emi as a feature
        input_df_clf_processed['max_monthly_emi'] = predicted_max_monthly_emi

        # Calculate engineered features (total_monthly_expenses and EMI_to_Salary_Ratio)
        input_df_clf_processed['total_monthly_expenses'] = input_data['monthly_rent'] + input_data['school_fees'] + input_data['college_fees'] + input_data['travel_expenses'] + input_data['groceries_utilities'] + input_data['other_monthly_expenses']
        input_df_clf_processed['EMI_to_Salary_Ratio'] = (input_data['current_emi_amount'] + predicted_max_monthly_emi) / input_data['monthly_salary'] if input_data['monthly_salary'] != 0 else 0

        # Ensure columns match training data for classification
        missing_cols_clf = set(classification_features_app) - set(input_df_clf_processed.columns)
        for c in missing_cols_clf:
            input_df_clf_processed[c] = 0
        input_df_clf_processed = input_df_clf_processed[classification_features_app]

        # Apply the loaded scaler to the appropriate numerical columns for classification
        numerical_cols_for_scaling_clf = [col for col in classification_features_app if col in ['age', 'monthly_salary', 'years_of_employment', 'monthly_rent',
                                                                                            'school_fees', 'college_fees', 'travel_expenses', 'groceries_utilities',
                                                                                            'other_monthly_expenses', 'current_emi_amount', 'credit_score', 'bank_balance',
                                                                                            'emergency_fund', 'requested_amount', 'max_monthly_emi',
                                                                                            'EMI_to_Salary_Ratio', 'total_monthly_expenses']]

        cols_to_scale_clf_exist = [col for col in numerical_cols_for_scaling_clf if col in input_df_clf_processed.columns]

        if cols_to_scale_clf_exist:
            input_df_clf_processed_scaled_part = scaler.transform(input_df_clf_processed[cols_to_scale_clf_exist])
            input_df_clf_processed[cols_to_scale_clf_exist] = input_df_clf_processed_scaled_part

        # --- Predict EMI Eligibility (Classification Model) ---
        predicted_eligibility_encoded = tuned_xgb_model.predict(input_df_clf_processed)
        predicted_eligibility = label_encoder.inverse_transform(predicted_eligibility_encoded)[0]


    # --- Streamlit UI: Results Display ---
    st.markdown("## 🔮 Prediction Results")
    
    col_emi, col_eligibility = st.columns(2)
    
    with col_emi:
        st.metric(
            label="Max Monthly EMI Affordability",
            value=f"₹{predicted_max_monthly_emi:.2f}",
            delta="Predicted by Regression Model"
        )
        st.caption("This is the maximum EMI the applicant is predicted to be able to pay monthly.")
    
    with col_eligibility:
        if predicted_eligibility == 'Eligible':
            st.markdown(f"""
            <div style="background-color: #d4edda; color: #155724; padding: 20px; border-radius: 10px; border: 2px solid #c3e6cb;">
                <h3 style="margin-top:0;">✅ EMI Eligibility: Eligible</h3>
                <p>The applicant is predicted to be **eligible** for the requested loan.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color: #f8d7da; color: #721c24; padding: 20px; border-radius: 10px; border: 2px solid #f5c6cb;">
                <h3 style="margin-top:0;">🛑 EMI Eligibility: Not Eligible</h3>
                <p>The applicant is predicted to be **not eligible** for the requested loan based on the predicted maximum affordability and other factors.</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("💡 Key Financial Metrics Used in Analysis")
    
    # Calculate key metrics to show the user
    total_monthly_expenses = input_df_clf_processed['total_monthly_expenses'].iloc[0] if 'total_monthly_expenses' in input_df_clf_processed.columns else 'N/A'
    current_debt_ratio = input_df_clf_processed['EMI_to_Salary_Ratio'].iloc[0] * 100 if 'EMI_to_Salary_Ratio' in input_df_clf_processed.columns else 'N/A'
    
    col_kpi_1, col_kpi_2, col_kpi_3 = st.columns(3)
    
    col_kpi_1.metric("Total Monthly Expenses", f"₹{total_monthly_expenses:.2f}" if isinstance(total_monthly_expenses, (float, int)) else total_monthly_expenses)
    col_kpi_2.metric("Current Loan EMI", f"₹{input_data['current_emi_amount']:.2f}")
    col_kpi_3.metric("Total EMI to Salary Ratio (Including New Loan)", f"{current_debt_ratio:.2f}%" if isinstance(current_debt_ratio, (float, int)) else current_debt_ratio)
