import pandas as pd
import numpy as np
import os
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

def run_pipeline():
    print("🚀 Starting Advanced Data Pipeline...")

    # ==========================================
    # 1. DOWNLOAD THE RAW DATA
    # ==========================================
    # 🚨 DON'T FORGET YOUR SHEET ID!
    SHEET_ID = 'YOUR_RAW_SHEET_ID_HERE' 
    url = f'https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet=Sheet1'
    
    try:
        df = pd.read_csv(url)
        print(f"✅ Successfully downloaded {len(df)} rows.")
    except Exception as e:
        print(f"❌ Failed to download raw data: {e}")
        return

    # ==========================================
    # 2. DATA CLEANING & PREPARATION
    # ==========================================
    print("🧹 Cleaning data...")
    
    financial_cols = ['Monthly Charge', 'Total Charges', 'Total Refunds', 'Total Revenue', 'Total Long Distance Charges']
    for col in financial_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    if 'Tenure in Months' in df.columns:
        df['Tenure in Months'] = pd.to_numeric(df['Tenure in Months'], errors='coerce').fillna(1)
        df['Tenure in Months'] = np.maximum(1, df['Tenure in Months'])

    # ==========================================
    # 3. CHURN VALUE IDENTIFICATION
    # ==========================================
    print("🏗️ Building base columns...")
    
    if 'Interaction Frequency (Annual)' not in df.columns:
        np.random.seed(42) 
        df['Interaction Frequency (Annual)'] = np.random.randint(0, 25, size=len(df))
        
    # Derive Churn Value securely
    if 'Customer Status' in df.columns:
        clean_status = df['Customer Status'].astype(str).str.strip().str.lower()
        df['Churn Value'] = np.where(clean_status == 'churned', 1, 0)
    elif 'Churn Label' in df.columns:
        clean_label = df['Churn Label'].astype(str).str.strip().str.lower()
        df['Churn Value'] = np.where(clean_label.isin(['yes', '1', 'true', 'y']), 1, 0)
    else:
        df['Churn Value'] = 0

    # ==========================================
    # 4. ADVANCED FEATURE ENGINEERING
    # ==========================================
    print("⚙️ Engineering advanced features...")

    # Calculate profitability
    df['Interaction Velocity (Per Month)'] = df['Interaction Frequency (Annual)'] / 12.0
    df['Estimated Lifetime Interactions'] = df['Interaction Velocity (Per Month)'] * df['Tenure in Months']
    df['Estimated Cost-to-Serve'] = df['Estimated Lifetime Interactions'] * 15.0 
    df['Net Customer Profitability'] = df['Total Revenue'] - df['Total Refunds'] - df['Estimated Cost-to-Serve']

    # Calculate Service Bundles
    service_cols = ['Phone Service', 'Multiple Lines', 'Internet Service', 'Online Security', 
                    'Online Backup', 'Device Protection Plan', 'Premium Tech Support', 
                    'Streaming TV', 'Streaming Movies', 'Streaming Music']
    available_services = [col for col in service_cols if col in df.columns]
    
    if available_services:
        df['Service Bundle Count'] = df[available_services].apply(lambda x: x.isin(['Yes', 1, '1']).sum(), axis=1)
    else:
        df['Service Bundle Count'] = 0

    # Calculate Risk Score
    if 'Contract' in df.columns:
        base_risk = np.where(df['Contract'] == 'Month-to-Month', 60, np.where(df['Contract'] == 'One Year', 30, 10))
        tenure_penalty = np.maximum(0, (24 - df['Tenure in Months'])) * 1.5
        df['Churn Risk Score'] = np.clip(base_risk + tenure_penalty + np.random.normal(0, 5, len(df)), 1, 99)
    else:
        df['Churn Risk Score'] = 50

    # ==========================================
    # 5. CUSTOMER VALUE SEGMENTATION (THE FIX)
    # ==========================================
    print("📊 Segmenting customers...")
    
    # Calculate dynamic thresholds based on the actual data distribution
    prof_top_30 = df['Net Customer Profitability'].quantile(0.70)
    prof_top_60 = df['Net Customer Profitability'].quantile(0.40)
    risk_top_30 = df['Churn Risk Score'].quantile(0.70)

    # REVISED RULES: No longer filtering by "Churn Value == 0" for standard segments!
    conditions_seg = [
        # 1. Regrettable Churn: They actually churned, had high tenure (4+ yrs), and multiple services
        (df['Churn Value'] == 1) & (df['Tenure in Months'] >= 48) & (df['Service Bundle Count'] >= 3),
        
        # 2. At risk Premium: High risk score AND decent profitability/services
        (df['Churn Risk Score'] >= risk_top_30) & ((df['Net Customer Profitability'] >= prof_top_60) | (df['Service Bundle Count'] >= 3)),
        
        # 3. VIP: Top 30% of profitability overall (regardless of current risk)
        (df['Net Customer Profitability'] >= prof_top_30),
        
        # 4. Upsell opportunity: Lower services, but stable tenure
        (df['Service Bundle Count'] <= 2) & (df['Tenure in Months'] > 12)
    ]
    
    choices_seg = ['Regrettable churn', 'At risk Premium', 'VIP', 'Upsell opportunity']
    df['Customer Value Segment'] = np.select(conditions_seg, choices_seg, default='Other')

    # DEBUG TRACKER
    print("\n--- SEGMENTATION RESULTS ---")
    print(df['Customer Value Segment'].value_counts())
    print("----------------------------\n")

    # ==========================================
    # 6. FINAL CLEANUP & UPLOAD
    # ==========================================
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    output_filename = 'Model_Ready_Data.csv'
    df.to_csv(output_filename, index=False)
    
    TARGET_FILE_ID = '1m7RHqafoVen_AKSXSKm69lituXBGj2XcD96gR-w63-E'

    try:
        creds_json = os.environ.get('GCP_CREDENTIALS')
        if creds_json:
            creds_dict = json.loads(creds_json)
            credentials = service_account.Credentials.from_service_account_info(
                creds_dict, scopes=['https://www.googleapis.com/auth/drive']
            )
            service = build('drive', 'v3', credentials=credentials)
            media = MediaFileUpload(output_filename, mimetype='text/csv', resumable=True)

            updated_file = service.files().update(fileId=TARGET_FILE_ID, media_body=media).execute()
            print(f"✅ SUCCESS! Uploaded to Drive. ID: {updated_file.get('id')}")
    except Exception as e:
        print(f"❌ Failed to upload to Google Drive: {e}")

if __name__ == '__main__':
    run_pipeline()
