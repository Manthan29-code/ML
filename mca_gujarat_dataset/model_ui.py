import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="MCA Gujarat Status Predictor", 
    layout="wide", 
    page_icon="🏢",
    initial_sidebar_state="expanded"
)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "LogisticRegression_pipeline.joblib")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

# --- APP UI ---
st.title("🏢 MCA Gujarat Company Status Predictor")
st.markdown("Predict the operational status of a company based on its financial and categorical features using the trained **Logistic Regression** model.")
st.divider()

if model is None:
    st.error("⚠️ Model file not found. Ensure `LogisticRegression_pipeline.joblib` exists in the same directory as this script.")
    st.stop()

# --- INPUT FORM ---
with st.form("prediction_form"):
    st.subheader("Company Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Classification & Industry**")
        company_class = st.selectbox("Company Class", ['Private', 'Public', 'Private(One Person Company)'])
        company_cat = st.selectbox("Company Category", ['Company limited by Shares', 'Company Limited by Guarantee', 'Unlimited Company'])
        company_subcat = st.selectbox("Company Sub-Category", ['Non-govt company', 'State Govt company', 'Subsidiary of Foreign Company', 'Guarantee and Association comp', 'Union Govt company'])
        business_activity = st.selectbox("Principal Business Activity", [
            'Manufacturing (Metals & Chemicals, and products thereof)', 
            'Business Services', 'Trading', 'Construction', 
            'Manufacturing (Machinery & Equipments)', 'Manufacturing (Textiles)', 
            'Finance', 'Community, personal & Social Services', 
            'Manufacturing (Food stuffs)', 'Transport, storage and Communications'
        ])
    
    with col2:
        st.markdown("**Financials**")
        auth_cap = st.number_input("Authorized Capital (INR)", min_value=0.0, value=500000.0, step=10000.0)
        paidup_cap = st.number_input("Paid-Up Capital (INR)", min_value=0.0, value=100000.0, step=10000.0)
        
        # Internal derived features shown to user
        ratio = paidup_cap / auth_cap if auth_cap > 0 else 0.0
        is_paidup = 1 if paidup_cap >= auth_cap else 0
        
        st.info(f"📊 **Paidup to Auth Ratio:** {ratio:.4f}")
        st.info(f"✅ **Fully Paid-Up Flag:** {'Yes' if is_paidup else 'No'}")
        
    with col3:
        st.markdown("**Registration & Age**")
        reg_year = st.number_input("Registration Year", min_value=1800, max_value=2050, value=2015, step=1)
        reg_month = st.selectbox("Registration Month", ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
        email_domain = st.text_input("Email Domain", value="gmail.com")
        
        company_age = 2026 - reg_year  # Base year inferred from dataset processing

    # Submit button
    submitted = st.form_submit_button("Predict Status 🚀", type="primary", use_container_width=True)

# --- PREDICTION LOGIC ---
if submitted:
    # Build dataframe containing all expected inputs strictly mimicking data preprocessing steps
    input_dict = {
        'COMPANY_CLASS': company_class,
        'COMPANY_CATEGORY': company_cat,
        'COMPANY_SUB_CATEGORY': company_subcat,
        'AUTHORIZED_CAP': auth_cap,
        'PAIDUP_CAPITAL': paidup_cap,
        'PRINCIPAL_BUSINESS_ACTIVITY_AS_PER_CIN': business_activity,
        'Registration_Year': reg_year,
        'Registration_Month': reg_month,
        'Company_Age_Years': company_age,
        'Paidup_to_Authorized_Ratio': ratio,
        'Is_Fully_PaidUp': is_paidup,
        'Email_Domain': email_domain,
        # Feature Engineering columns generated in the Jupyter logic
        'AUTHORIZED_CAP_log': np.log1p(auth_cap),
        'PAIDUP_CAPITAL_log': np.log1p(paidup_cap),
        'Paidup_to_Authorized_Ratio_log': np.log1p(ratio)
    }
    
    input_df = pd.DataFrame([input_dict])
    
    with st.spinner("Processing Model Prediction..."):
        try:
            # Predict
            pred = model.predict(input_df)[0]
            
            st.success(f"### 🎉 Predicted Company Status: **{pred}**")
            
            # Show probabilities if model supports it (Logistic Regression does)
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_df)[0]
                classes = model.classes_
                
                st.write("**Prediction Probabilities:**")
                prob_df = pd.DataFrame({'Status': classes, 'Probability': proba}).sort_values(by='Probability', ascending=False)
                
                # Render clean styled dataframe
                st.dataframe(
                    prob_df.style.format({'Probability': '{:.2%}'}).background_gradient(cmap='Greens', subset=['Probability']),
                    hide_index=True,
                    use_container_width=True
                )
                
        except Exception as e:
            st.error(f"Prediction failed. Error message: {e}")
            st.warning("Please verify that the raw inputs entered exactly match the columns present during your fit() phase.")