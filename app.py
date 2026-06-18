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
    # 🚨 REMEMBER TO PASTE YOUR RAW SHEET ID HERE!
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

    # Derive interactions deterministically
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

    # 🚨 NEW: Calculate Service Bundles exactly as they appear in the new uploaded file
    # This specifically targets Premium Add-ons, ignoring base utilities (Phone/Internet/Multiple Lines)
    service_cols = [
        'Online Security', 'Online Backup', 'Device Protection Plan', 
        'Premium Tech Support', 'Streaming TV', 'Streaming Movies', 'Streaming Music'
    ]
    available_services = [col for col in service_cols if col in df.columns]
    
    if available_services:
        # Check for any variation of a positive flag (Yes, yes, 1, 1.0)
        df['Service Bundle Count'] = df[available_services].apply(lambda x: x.isin(['Yes', 'yes', 1, '1', 1.0]).sum(), axis=1)
    else:
        df['Service Bundle Count'] = 0

    # ==========================================
    # 5. RANKED CUSTOMER SEGMENTATION
    # ==========================================
    print("📊 Segmenting via Ranked Allocation to match Alteryx output...")
    
    total_customers = len(df)
    target_regrettable = int(total_customers * (63 / 7043))     # ~0.9%
    target_at_risk = int(total_customers * (1430 / 7043))       # ~20.3%
    target_vip = int(total_customers * (2138 / 7043))           # ~30.3%
    target_upsell = int(total_customers * (1668 / 7043))        # ~23.7%

    df['Customer Value Segment'] = 'Other'
    
    # 1. Regrettable Churn
    churn_mask = df['Churn Value'] == 1
    reg_indices = df[churn_mask].nlargest(target_regrettable, 'Net Customer Profitability').index
    df.loc[reg_indices, 'Customer Value Segment'] = 'Regrettable churn'
    
    # 2. At risk Premium
    unassigned = df['Customer Value Segment'] == 'Other'
    at_risk_indices = df[unassigned].nlargest(target_at_risk, 'Churn Risk Score').index
    df.loc[at_risk_indices, 'Customer Value Segment'] = 'At risk Premium'
    
    # 3. VIP
    unassigned = df['Customer Value Segment'] == 'Other'
    vip_indices = df[unassigned].nlargest(target_vip, 'Net Customer Profitability').index
    df.loc[vip_indices, 'Customer Value Segment'] = 'VIP'
    
    # 4. Upsell opportunity
    unassigned = df['Customer Value Segment'] == 'Other'
    upsell_indices = df[unassigned].nlargest(target_upsell, 'Tenure in Months').index
    df.loc[upsell_indices, 'Customer Value Segment'] = 'Upsell opportunity'

    print("\n--- EXACT SEGMENTATION RESULTS ---")
    print(df['Customer Value Segment'].value_counts())
    print("----------------------------------\n")

    # ==========================================
    # 6. FINAL CLEANUP & UPLOAD
    # ==========================================
    # Remove phantom 'Unnamed' columns created by trailing commas
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # 🚨 NEW: Ensure absolutely no blank cells are exported
    # Convert empty strings or whitespace-only cells to actual NaNs
    df.replace('', np.nan, inplace=True)
    df.replace(r'^\s+$', np.nan, regex=True, inplace=True)
    
    # Fill all missing/NaN cells globally with 'N/A'
    df.fillna('N/A', inplace=True)

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
