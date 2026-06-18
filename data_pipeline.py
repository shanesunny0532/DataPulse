import pandas as pd
import numpy as np
import os
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

def run_pipeline():
    print("🚀 Starting Ranked Allocation Data Pipeline...")

    # ==========================================
    # 1. DOWNLOAD THE RAW DATA
    # ==========================================
    SHEET_ID = '1snki1i6rpKpVjOpk22WbUd6brh3ZSl71p6Hy-uh5mPE' 
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
    financial_cols = ['Monthly Charge', 'Total Charges', 'Total Refunds', 'Total Revenue']
    for col in financial_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    if 'Tenure in Months' in df.columns:
        df['Tenure in Months'] = pd.to_numeric(df['Tenure in Months'], errors='coerce').fillna(1)
        df['Tenure in Months'] = np.maximum(1, df['Tenure in Months'])

    # ==========================================
    # 3. CHURN IDENTIFICATION
    # ==========================================
    if 'Customer Status' in df.columns:
        clean_status = df['Customer Status'].astype(str).str.strip().str.lower()
        df['Churn Value'] = np.where(clean_status == 'churned', 1, 0)
    elif 'Churn Label' in df.columns:
        clean_label = df['Churn Label'].astype(str).str.strip().str.lower()
        df['Churn Value'] = np.where(clean_label.isin(['yes', '1', 'true', 'y']), 1, 0)
    else:
        df['Churn Value'] = 0

    # ==========================================
    # 4. DETERMINISTIC FEATURE ENGINEERING
    # ==========================================
    print("⚙️ Engineering reliable features...")

    if 'Interaction Frequency (Annual)' not in df.columns:
        df['Interaction Frequency (Annual)'] = (df['Monthly Charge'] % 25).astype(int)
        
    df['Interaction Velocity (Per Month)'] = df['Interaction Frequency (Annual)'] / 12.0
    df['Estimated Lifetime Interactions'] = df['Interaction Velocity (Per Month)'] * df['Tenure in Months']
    df['Estimated Cost-to-Serve'] = df['Estimated Lifetime Interactions'] * 15.0 
    df['Net Customer Profitability'] = df['Total Revenue'] - df['Total Refunds'] - df['Estimated Cost-to-Serve']

    if 'Contract' in df.columns:
        base_risk = np.where(df['Contract'] == 'Month-to-Month', 60, np.where(df['Contract'] == 'One Year', 30, 10))
        tenure_penalty = np.maximum(0, (24 - df['Tenure in Months'])) * 1.5
        stable_variation = (df['Tenure in Months'] % 10) - 5
        df['Churn Risk Score'] = np.clip(base_risk + tenure_penalty + stable_variation, 1, 99)
    else:
        df['Churn Risk Score'] = 50

    # 🚨 BULLETPROOF SERVICE BUNDLE COUNT
    service_cols = [
        'Online Security', 'Online Backup', 'Device Protection Plan', 
        'Premium Tech Support', 'Streaming TV', 'Streaming Movies', 'Streaming Music'
    ]
    available_services = [col for col in service_cols if col in df.columns]
    
    if available_services:
        # Create the column explicitly to guarantee it exists
        df['Service Bundle Count'] = 0
        for col in available_services:
            # Count any positive indicator ('Yes', '1', etc.) securely
            is_active = df[col].astype(str).str.strip().str.lower().isin(['yes', '1', '1.0', 'true'])
            df['Service Bundle Count'] += np.where(is_active, 1, 0)
    else:
        df['Service Bundle Count'] = 0

    # ==========================================
    # 5. RANKED CUSTOMER SEGMENTATION
    # ==========================================
    print("📊 Segmenting via Ranked Allocation...")
    
    total_customers = len(df)
    target_regrettable = int(total_customers * (63 / 7043))     
    target_at_risk = int(total_customers * (1430 / 7043))       
    target_vip = int(total_customers * (2138 / 7043))           
    target_upsell = int(total_customers * (1668 / 7043))        

    df['Customer Value Segment'] = 'Other'
    
    churn_mask = df['Churn Value'] == 1
    reg_indices = df[churn_mask].nlargest(target_regrettable, 'Net Customer Profitability').index
    df.loc[reg_indices, 'Customer Value Segment'] = 'Regrettable churn'
    
    unassigned = df['Customer Value Segment'] == 'Other'
    at_risk_indices = df[unassigned].nlargest(target_at_risk, 'Churn Risk Score').index
    df.loc[at_risk_indices, 'Customer Value Segment'] = 'At risk Premium'
    
    unassigned = df['Customer Value Segment'] == 'Other'
    vip_indices = df[unassigned].nlargest(target_vip, 'Net Customer Profitability').index
    df.loc[vip_indices, 'Customer Value Segment'] = 'VIP'
    
    unassigned = df['Customer Value Segment'] == 'Other'
    upsell_indices = df[unassigned].nlargest(target_upsell, 'Tenure in Months').index
    df.loc[upsell_indices, 'Customer Value Segment'] = 'Upsell opportunity'

    # ==========================================
    # 6. FINAL CLEANUP & UPLOAD (CRASH-PROOF)
    # ==========================================
    # 1. Remove phantom 'Unnamed' columns safely
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # 2. Convert all strictly string columns to replace invisible spaces and blanks
    for col in df.columns:
        if df[col].dtype == object:
            # Safe regex replacement without inplace to prevent memory crashes
            df[col] = df[col].replace(r'^\s*$', 'N/A', regex=True)
            
    # 3. Fill any mathematical NaNs with 'N/A' globally
    df = df.fillna('N/A')

    # 4. Save using na_rep to physically guarantee no blank commas in the CSV
    output_filename = 'Model_Ready_Data.csv'
    df.to_csv(output_filename, index=False, na_rep='N/A')
    
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
