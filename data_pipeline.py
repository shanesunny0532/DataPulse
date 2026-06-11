import pandas as pd
import numpy as np

def run_pipeline():
    print("🚀 Starting Advanced Data Pipeline...")

    # 1. DOWNLOAD THE RAW DATA
    # Replace 'YOUR_SHEET_ID' with the ID from your Google Sheet URL
    SHEET_ID = '1snki1i6rpKpVjOpk22WbUd6brh3ZSl71p6Hy-uh5mPE'
    SHEET_NAME = 'Sheet1'
    url = f'https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}'
    
    try:
        # If testing locally, you can swap the url for your raw CSV file path
        df = pd.read_csv(url)
        print(f"✅ Successfully downloaded {len(df)} rows.")
    except Exception as e:
        print(f"❌ Failed to download data: {e}")
        return

    # ==========================================
    # 2. DATA CLEANING & PREPARATION
    # ==========================================
    print("🧹 Cleaning data...")
    
    if 'Internet Type' in df.columns:
        df['Internet Type'] = df['Internet Type'].replace(r'^\s*$', np.nan, regex=True).fillna('None')

    # Ensure all financial columns are numeric
    financial_cols = ['Monthly Charge', 'Total Charges', 'Total Refunds', 'Total Revenue', 'Total Long Distance Charges']
    for col in financial_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Ensure Tenure is numeric and at least 1 to avoid division by zero
    if 'Tenure in Months' in df.columns:
        df['Tenure in Months'] = pd.to_numeric(df['Tenure in Months'], errors='coerce').fillna(1)
        df['Tenure in Months'] = np.maximum(1, df['Tenure in Months'])

    # Ensure Interaction Frequency exists (Simulate if it's missing from raw data)
    if 'Interaction Frequency (Annual)' not in df.columns:
        np.random.seed(42)
        df['Interaction Frequency (Annual)'] = np.random.randint(0, 25, size=len(df))

    # ==========================================
    # 3. ADVANCED FEATURE ENGINEERING
    # ==========================================
    print("⚙️ Engineering advanced features and segments...")

    # A. INTERACTION METRICS
    df['Interaction Velocity (Per Month)'] = df['Interaction Frequency (Annual)'] / 12.0
    df['Interaction Velocity'] = df['Interaction Frequency (Annual)'] / df['Tenure in Months']
    df['Estimated Lifetime Interactions'] = df['Interaction Velocity (Per Month)'] * df['Tenure in Months']
    
    # B. REVENUE & COST METRICS
    if 'Total Revenue' in df.columns:
        df['Avg Revenue Per Month'] = df['Total Revenue'] / df['Tenure in Months']
        df['Revenue-to-Interaction Ratio'] = df['Total Revenue'] / np.maximum(1, df['Estimated Lifetime Interactions'])
        df['Refund Rate (%)'] = (df['Total Refunds'] / np.maximum(1, df['Total Revenue'])) * 100
        df['Refund Rate (%)'] = df['Refund Rate (%)'].round(2)
        
    df['Estimated Cost-to-Serve'] = df['Estimated Lifetime Interactions'] * 15.0  # Assuming $15 cost per interaction
    df['Net Customer Profitability'] = df['Total Revenue'] - df['Total Refunds'] - df['Estimated Cost-to-Serve']

    if 'Total Long Distance Charges' in df.columns:
        df['Avg Monthly Long Distance Charges'] = df['Total Long Distance Charges'] / df['Tenure in Months']

    # C. CUSTOMER STATUS & TENURE GROUPS
    if 'Churn Value' in df.columns:
        conditions = [
            (df['Tenure in Months'] <= 1),
            (df['Churn Value'] == 1),
            (df['Churn Value'] == 0)
        ]
        choices = ['Joined', 'Churned', 'Stayed']
        df['Customer Status'] = np.select(conditions, choices, default='Unknown')

    bins = [0, 6, 12, 24, 48, 72, 1000]
    labels = ['Months 0-6', 'Months 7-12', 'Months 13-24', 'Months 25-48', 'Months 49-72', 'Months 73+']
    df['Tenure Group'] = pd.cut(df['Tenure in Months'], bins=bins, labels=labels, right=True)

    # D. SERVICE BUNDLES & ENGAGEMENT
    service_cols = ['Phone Service', 'Multiple Lines', 'Internet Service', 'Online Security', 
                    'Online Backup', 'Device Protection Plan', 'Premium Tech Support', 
                    'Streaming TV', 'Streaming Movies', 'Streaming Music']
    
    available_services = [col for col in service_cols if col in df.columns]
    if available_services:
        df['Service Bundle Count'] = df[available_services].apply(lambda x: x.isin(['Yes', 1, '1']).sum(), axis=1)
        df['Revenue Per Service'] = df['Monthly Charge'] / np.maximum(1, df['Service Bundle Count'])
        
        # Calculate a normalized Engagement Score (0 to 1 scale)
        max_services = len(available_services)
        service_score = df['Service Bundle Count'] / max_services
        df['Engagement Score'] = (service_score * 0.6) + (df['Tenure in Months'] / 72 * 0.4)
        df['Engagement Score'] = df['Engagement Score'].round(3)

    # E. RISK FLAGS & SCORES
    if 'Contract' in df.columns:
        df['Contract Risk Flag'] = np.where((df['Contract'] == 'Month-to-Month') & (df['Tenure in Months'] <= 12), 1, 0)

    # Normalize Profitability Score (0 to 100)
    min_prof = df['Net Customer Profitability'].min()
    max_prof = df['Net Customer Profitability'].max()
    df['Profitability Score'] = ((df['Net Customer Profitability'] - min_prof) / (max_prof - min_prof)) * 100

    # Simulate a baseline Churn Risk Score if an ML model isn't actively attached
    # (Higher risk for M2M contracts, low tenure, and low engagement)
    if 'Contract' in df.columns:
        base_risk = np.where(df['Contract'] == 'Month-to-Month', 60, np.where(df['Contract'] == 'One Year', 30, 10))
        tenure_penalty = np.maximum(0, (24 - df['Tenure in Months'])) * 1.5
        df['Churn Risk Score'] = np.clip(base_risk + tenure_penalty + np.random.normal(0, 5, len(df)), 1, 99)

    df['Retention Priority Score'] = (df['Churn Risk Score'] * 0.6) + (df['Profitability Score'] * 0.4)

    # F. CUSTOMER VALUE SEGMENTATION
    # Reproducing the exact segments: VIP, Upsell opportunity, At risk Premium, Regrettable churn, Other
    conditions_seg = [
        (df['Net Customer Profitability'] > df['Net Customer Profitability'].quantile(0.8)) & (df['Churn Risk Score'] < 40),
        (df['Net Customer Profitability'] > df['Net Customer Profitability'].quantile(0.7)) & (df['Churn Risk Score'] >= 60),
        (df['Net Customer Profitability'] > df['Net Customer Profitability'].quantile(0.8)) & (df['Churn Value'] == 1),
        (df['Service Bundle Count'] <= 2) & (df['Tenure in Months'] > 12) & (df['Churn Risk Score'] < 50)
    ]
    choices_seg = ['VIP', 'At risk Premium', 'Regrettable churn', 'Upsell opportunity']
    df['Customer Value Segment'] = np.select(conditions_seg, choices_seg, default='Other')

    # Flag High Value Customers
    df['High Value Customer'] = np.where(df['Customer Value Segment'].isin(['VIP', 'At risk Premium']), 1, 0)

    # 4. SAVE THE FINAL DATASET LOCALLY FIRST
    output_filename = 'Model_Ready_Data.csv'
    df.to_csv(output_filename, index=False)
    print("✅ Data cleaned and saved locally. Preparing to upload to Google Drive...")

   # 5. UPLOAD TO GOOGLE DRIVE (Direct File Update)
    import os
    import json
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    # Paste the ID of your specific placeholder file here
    TARGET_FILE_ID = '1m7RHqafoVen_AKSXSKm69lituXBGj2XcD96gR-w63-E'

    try:
        # Load the secret credentials passed from GitHub Actions
        creds_json = os.environ.get('GCP_CREDENTIALS')
        if not creds_json:
            raise ValueError("GCP_CREDENTIALS environment variable not found!")
            
        creds_dict = json.loads(creds_json)
        credentials = service_account.Credentials.from_service_account_info(
            creds_dict, scopes=['https://www.googleapis.com/auth/drive']
        )
