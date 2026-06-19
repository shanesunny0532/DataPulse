import streamlit as st
import pandas as pd
import joblib

# 1. Page Configuration
st.set_page_config(page_title="Churn Predictor", layout="wide")
st.title("DataPulse: Telecom Churn Predictor 📊")
st.markdown("Enter customer details below to instantly calculate their churn risk.")

# 2. Load the Model and Feature List
@st.cache_resource # This ensures the model only loads once, keeping the app fast
def load_model():
    model = joblib.load('telecom_rf_model_lite.pkl')
    features = joblib.load('model_features_lite.pkl')
    return model, features

rf_model, model_features = load_model()

# 3. Build the User Input Form (The Sidebar)
st.sidebar.header("Customer Profile")

# Create dropdowns and sliders for the user to input data
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure in Months", 1, 72, 12)
monthly_charge = st.sidebar.number_input("Monthly Charge ($)", min_value=10.0, value=50.0)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])
internet = st.sidebar.selectbox("Internet Service", ["Fiber Optic", "DSL", "None"])

# 4. Format the Data for Prediction
if st.button("Predict Churn Risk"):
    
    # Put the user inputs into a dictionary
    input_data = {
        'Gender': [gender],
        'Senior Citizen': [1 if senior == "Yes" else 0],
        'Tenure in Months': [tenure],
        'Monthly Charge': [monthly_charge],
        'Contract': [contract],
        'Internet Type': [internet],
        # Note: In a real app, you would add inputs for ALL columns your model needs.
        # We add 0s for missing ones during the encoding step below.
    }
    
    input_df = pd.DataFrame(input_data)
    
    # 5. One-Hot Encode to match the training data
    encoded_input = pd.get_dummies(input_df)
    
    # Crucial Step: Ensure the input has the EXACT same columns as the training data
    encoded_input = encoded_input.reindex(columns=model_features, fill_value=0)
    
    # 6. Make the Prediction
    prediction = rf_model.predict_proba(encoded_input)
    churn_risk = prediction[0][1] * 100
    
    # 7. Display the Results beautifully
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="Predicted Churn Risk", value=f"{churn_risk:.1f}%")
        
    with col2:
        if churn_risk >= 75:
            st.error("🚨 High Risk Customer - Immediate Action Required")
        elif churn_risk >= 40:
            st.warning("⚠️ Medium Risk Customer - Monitor Closely")
        else:
            st.success("✅ Low Risk Customer - Safe")
