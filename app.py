import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Churn Predictor Pro", layout="wide")
st.title("DataPulse: Advanced Churn Predictor 📊")

@st.cache_resource 
def load_model():
    model = joblib.load('telecom_rf_model_lite.pkl')
    features = joblib.load('model_features_lite.pkl')
    return model, features

rf_model, model_features = load_model()

st.sidebar.title("Customer Configuration")

# --- SECTION 1: DEMOGRAPHICS ---
with st.sidebar.expander("👤 Demographics & Account", expanded=True):
    age = st.slider("Age", 18, 90, 45)
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    dependents = st.number_input("Number of Dependents", 0, 5, 0)
    referrals = st.slider("Number of Referrals", 0, 10, 0)

# --- SECTION 2: CONTRACT & BILLING ---
with st.sidebar.expander("💳 Contract & Billing", expanded=True):
    contract = st.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])
    tenure = st.slider("Tenure in Months", 1, 72, 12)
    monthly_charge = st.number_input("Monthly Charge ($)", min_value=10.0, max_value=200.0, value=85.0)
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

# --- SECTION 3: SERVICES & USAGE ---
with st.sidebar.expander("🌐 Services & Usage", expanded=False):
    internet = st.selectbox("Internet Service", ["Fiber Optic", "DSL", "None"])
    gb_download = st.slider("Avg Monthly GB Download", 0, 100, 20)
    bundle_count = st.slider("Service Bundle Count", 0, 4, 2)
    tech_support = st.selectbox("Premium Tech Support", ["Yes", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No"])

# --- SECTION 4: BEHAVIOR & SENTIMENT ---
with st.sidebar.expander("😡 Behavior & Sentiment", expanded=True):
    satisfaction = st.slider("Satisfaction Score (1-5)", 1, 5, 3)
    engagement = st.slider("Engagement Score", 0.0, 1.0, 0.5)
    refund_rate = st.slider("Refund Rate (%)", 0.0, 100.0, 0.0)
    interaction_freq = st.slider("Interaction Frequency (Annual)", 0, 30, 2)

if st.button("Predict Churn Risk", type="primary", use_container_width=True):
    
    # Map all inputs to match the EXACT column names from the dataset
    input_data = {
        'Age': [age],
        'Senior Citizen': [1 if senior == "Yes" else 0],
        'Number of Dependents': [dependents],
        'Number of Referrals': [referrals],
        'Contract': [contract],
        'Tenure in Months': [tenure],
        'Monthly Charge': [monthly_charge],
        'Paperless Billing': [1 if paperless == "Yes" else 0],
        'Internet Type': [internet],
        'Avg Monthly GB Download': [gb_download],
        'Service Bundle Count': [bundle_count],
        'Premium Tech Support': [1 if tech_support == "Yes" else 0],
        'Online Security': [1 if online_security == "Yes" else 0],
        'Satisfaction Score': [satisfaction],
        'Engagement Score': [engagement],
        'Refund Rate (%)': [refund_rate],
        'Interaction Frequency (Annual)': [interaction_freq],
        # Auto-calculating a logical velocity based on frequency and tenure
        'Interaction Velocity': [interaction_freq / max(1, (tenure/12))] 
    }
    
    input_df = pd.DataFrame(input_data)
    
    # One-Hot Encode and align with the training columns
    encoded_input = pd.get_dummies(input_df)
    encoded_input = encoded_input.reindex(columns=model_features, fill_value=0)
    
    # Predict
    prediction = rf_model.predict_proba(encoded_input)
    churn_risk = prediction[0][1] * 100
    
    # Display Results
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="Predicted Churn Probability", value=f"{churn_risk:.1f}%")
        
    with col2:
        if churn_risk >= 50:
            st.error("🚨 **High Risk Customer** - Immediate Action Required")
            st.caption("Suggested Action: Route to Retention Team & Offer Promotional Discount.")
        elif churn_risk >= 25:
            st.warning("⚠️ **Medium Risk Customer** - Monitor Closely")
            st.caption("Suggested Action: Send Satisfaction Survey & Highlight Unused Features.")
        else:
            st.success("✅ **Low Risk Customer** - Safe")
            st.caption("Suggested Action: Routine Marketing & Upsell Opportunities.")
