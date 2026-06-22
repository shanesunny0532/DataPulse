import streamlit.components.v1 as components
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Churn Predictor", layout="wide")
st.title("DataPulse: Churn Predictor 📊")

@st.cache_resource 
def load_model():
    # 🚨 Updated to load the newly exported files!
    model = joblib.load('churn_rf_model.pkl')
    features = joblib.load('model_features.pkl')
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
    # Moved Refund Rate here since the Behavior section was removed
    refund_rate = st.slider("Refund Rate (%)", 0.0, 100.0, 0.0)

# --- SECTION 3: SERVICES & USAGE ---
with st.sidebar.expander("🌐 Services & Usage", expanded=False):
    internet = st.selectbox("Internet Service", ["Fiber Optic", "DSL", "None"])
    gb_download = st.slider("Avg Monthly GB Download", 0, 100, 20)
    bundle_count = st.slider("Service Bundle Count", 0, 4, 2)
    tech_support = st.selectbox("Premium Tech Support", ["Yes", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No"])

# Note: Behavior & Sentiment section removed because Satisfaction, Engagement, 
# and Interaction metrics were dropped from the model to prevent data leakage!

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
        'Refund Rate (%)': [refund_rate]
        # Satisfaction, Engagement, and Interaction metrics safely removed
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
        elif churn_risk >= 35:
            st.warning("⚠️ **Medium Risk Customer** - Monitor Closely")
            st.caption("Suggested Action: Send Satisfaction Survey & Highlight Unused Features.")
        else:
            st.success("✅ **Low Risk Customer** - Safe")
            st.caption("Suggested Action: Routine Marketing & Upsell Opportunities.")
            
# --- SECTION 5: ROI CALCULATOR (EMBEDDED HTML) ---
st.divider()
st.subheader("📊 Business ROI & Confusion Matrix Calculator")
st.write("Adjust the model's test numbers below to calculate the projected financial impact.")

html_calculator = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #333; margin: 0; padding: 10px; }
        .grid-container { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .card { background: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
        h3 { margin-top: 0; color: #2c3e50; border-bottom: 2px solid #e0e0e0; padding-bottom: 10px; font-size: 18px; }
        h4 { margin: 10px 0; font-size: 22px; }
        label { font-weight: bold; font-size: 14px; display: block; margin-top: 12px; }
        input { width: 100%; padding: 8px; margin-top: 4px; border: 1px solid #ccc; border-radius: 5px; font-size: 16px; box-sizing: border-box; }
        .metric { font-size: 20px; font-weight: bold; color: #0066cc; }
        .positive { color: #28a745; font-size: 20px; font-weight: bold; }
        .negative { color: #dc3545; font-size: 20px; font-weight: bold; }
        .highlight { color: #20c997; font-size: 28px; font-weight: 900; }
        hr { border: 0; border-top: 1px solid #ddd; margin: 25px 0; }
    </style>
</head>
<body>
    <div class="grid-container">
        <div class="card">
            <h3>1. Model Test Results</h3>
            <label>True Positives (Caught Churners):</label>
            <input type="number" id="tp" value="363" oninput="calculate()">
            <label>False Positives (False Alarms):</label>
            <input type="number" id="fp" value="36" oninput="calculate()">
            <label>False Negatives (Missed Churners):</label>
            <input type="number" id="fn" value="37" oninput="calculate()">
            <label>True Negatives (Happy Customers):</label>
            <input type="number" id="tn" value="973" oninput="calculate()">
            
            <hr>
            <h3>2. Business Strategy</h3>
            <label>Average Customer Lifetime Value ($):</label>
            <input type="number" id="ltv" value="1200" oninput="calculate()">
            <label>Cost of Retention Discount/Offer ($):</label>
            <input type="number" id="cost" value="100" oninput="calculate()">
            <label>Offer Acceptance Rate (%):</label>
            <input type="number" id="success" value="40" oninput="calculate()">
        </div>
        
        <div class="card">
            <h3>Model Performance Metrics</h3>
            <p>Recall (Caught %): <span class="metric" id="recall">0%</span></p>
            <p>Precision (Accuracy of Alarms): <span class="metric" id="precision">0%</span></p>
            <p>F1-Score: <span class="metric" id="f1">0%</span></p>
            
            <hr>
            <h3>Projected Campaign ROI</h3>
            <p>Gross Value of Saved Customers: $<span id="gross_saved" class="positive">0</span></p>
            <p>Wasted Cost on False Alarms: $<span id="wasted_cost" class="negative">0</span></p>
            <p>Valid Cost on True Churners: $<span id="valid_cost" class="negative">0</span></p>
            <br>
            <h4 style="color: #333;">Estimated Net ROI:</h4>
            <span class="highlight" id="net_roi">$0</span>
        </div>
    </div>

    <script>
        function calculate() {
            // Get values
            let tp = parseFloat(document.getElementById('tp').value) || 0;
            let fp = parseFloat(document.getElementById('fp').value) || 0;
            let fn = parseFloat(document.getElementById('fn').value) || 0;
            let tn = parseFloat(document.getElementById('tn').value) || 0;
            
            let ltv = parseFloat(document.getElementById('ltv').value) || 0;
            let cost = parseFloat(document.getElementById('cost').value) || 0;
            let success = (parseFloat(document.getElementById('success').value) || 0) / 100;
            
            // AI Metrics
            let recall = tp / (tp + fn);
            let precision = tp / (tp + fp);
            let f1 = 2 * (precision * recall) / (precision + recall);
            
            document.getElementById('recall').innerText = (recall * 100).toFixed(1) + '%';
            document.getElementById('precision').innerText = (precision * 100).toFixed(1) + '%';
            document.getElementById('f1').innerText = (f1 * 100).toFixed(1) + '%';
            
            // ROI Math
            let saved_customers = tp * success;
            let gross_saved = saved_customers * ltv;
            
            let wasted_cost = fp * cost; // Money spent on people who weren't leaving
            let valid_cost = tp * cost;  // Money spent trying to save real churners
            
            let net_roi = gross_saved - (wasted_cost + valid_cost);
            
            // Format Currency
            document.getElementById('gross_saved').innerText = gross_saved.toLocaleString(undefined, {minimumFractionDigits: 0, maximumFractionDigits: 0});
            document.getElementById('wasted_cost').innerText = wasted_cost.toLocaleString(undefined, {minimumFractionDigits: 0, maximumFractionDigits: 0});
            document.getElementById('valid_cost').innerText = valid_cost.toLocaleString(undefined, {minimumFractionDigits: 0, maximumFractionDigits: 0});
            
            let roi_el = document.getElementById('net_roi');
            roi_el.innerText = '$' + net_roi.toLocaleString(undefined, {minimumFractionDigits: 0, maximumFractionDigits: 0});
            roi_el.style.color = net_roi >= 0 ? '#20c997' : '#dc3545';
        }
        
        // Run on load
        window.onload = calculate;
    </script>
</body>
</html>
"""

components.html(html_calculator, height=650, scrolling=True)
