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
    SHEET_ID = '1snki1i6rpKpVjOpk22WbUd6brh3ZSl71p6Hy-uh5mPE'
    SHEET_NAME = 'Sheet1'
    url = f'https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}'
    
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
    
    if 'Internet Type' in df.columns:
        df['Internet Type'] = df['Internet Type'].replace(r'^\s*$', np.nan, regex=True).fillna('None')

    financial_cols = ['Monthly Charge', 'Total Charges', 'Total Refunds', 'Total Revenue', 'Total Long Distance Charges']
    for col in financial_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    if 'Tenure in Months' in df.columns:
        df['Tenure in Months'] = pd.to_numeric(df['Tenure in Months'], errors='coerce').fillna(1)
        df['Tenure in Months'] = np.maximum(1, df['Tenure in Months'])

    # ==========================================
    # 3. CREATE MISSING BASE COLUMNS (THE BULLDOZER FIX)
    # ==========================================
    print("🏗️ Building missing base columns for engineering...")
    
    if 'Interaction Frequency (Annual)' not in df.columns:
        np.random.seed(42) 
        df['Interaction Frequency (Annual)'] = np.random.randint(0, 25, size=len(df))
        
    # 🚨 Find the correct Churn column no matter what it's named
    target_churn_col = None
    if 'Churn Value' in df.columns:
        target_churn_col = 'Churn Value'
    elif 'Churn Label' in df.columns:
        target_churn_col = 'Churn Label'
    elif 'Churn' in df.columns:
        target_churn_col = 'Churn'

    # 🚨 Strip invisible spaces, lower-case it, and map directly
    if target_churn_col:
        clean_churn = df[target_churn_col].astype(str).str.strip().str.lower()
        # Anything that means 'Yes' becomes 1. Everything else becomes 0.
        df['Churn Value'] = clean_churn.map({'yes': 1, '1': 1, '1.0': 1, 'true': 1}).fillna(0)
    else:
        df['Churn Value'] = 0

    # ==========================================
    # 4. ADVANCED FEATURE ENGINEERING
    # ==========================================
    print("⚙️ Engineering advanced features and segments...")

    if 'Interaction Frequency (Annual)' in df.columns:
        df['Interaction Velocity (Per Month)'] = df['Interaction Frequency (Annual)'] / 12.0
        df['Interaction Velocity'] = df['Interaction Frequency (Annual)'] / df['Tenure in Months']
        df['Estimated Lifetime Interactions'] = df['Interaction Velocity (Per Month)'] * df['Tenure in Months']
    
    if 'Total Revenue' in df.columns and 'Estimated Lifetime Interactions' in df.columns:
        df['Avg Revenue Per Month'] = df['Total Revenue'] / df['Tenure in Months']
        df['Revenue-to-Interaction Ratio'] = df['Total Revenue'] / np.maximum(1, df['Estimated Lifetime Interactions'])
        df['Refund Rate (%)'] = (df['Total Refunds'] / np.maximum(1, df['Total Revenue'])) * 100
        df['Refund Rate (%)'] = df['Refund Rate (%)'].round(2)
        
        df['Estimated Cost-to-Serve'] = df['Estimated Lifetime Interactions'] * 15.0 
        df['Net Customer Profitability'] = df['Total Revenue'] - df['Total Refunds'] - df['Estimated Cost-to-Serve']

    if 'Total Long Distance Charges' in df.columns:
        df['Avg Monthly Long Distance Charges'] = df['Total Long Distance Charges'] / df['Tenure in Months']

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

    service_cols = ['Phone Service', 'Multiple Lines', 'Internet Service', 'Online Security', 
                    'Online Backup', 'Device Protection Plan', 'Premium Tech Support', 
                    'Streaming TV', 'Streaming Movies', 'Streaming Music']
    
    available_services = [col for col in service_cols if col in df.columns]
    if available_services:
        df['Service Bundle Count'] = df[available_services].apply(lambda x: x.isin(['Yes', 1, '1']).sum(), axis=1)
        df['Revenue Per Service'] = df['Monthly Charge'] / np.maximum(1, df['Service Bundle Count'])
        
        max_services = len(available_services)
        service_score = df['Service Bundle Count'] / max_services
        df['Engagement Score'] = (service_score * 0.6) + (df['Tenure in Months'] / 72 * 0.4)
        df['Engagement Score'] = df['Engagement Score'].round(3)

    if 'Contract' in df.columns:
        df['Contract Risk Flag'] = np.where((df['Contract'] == 'Month-to-Month') & (df['Tenure in Months'] <= 12), 1, 0)

    if 'Net Customer Profitability' in df.columns:
        min_prof = df['Net Customer Profitability'].min()
        max_prof = df['Net Customer Profitability'].max()
        if max_prof > min_prof:
            df['Profitability Score'] = ((df['Net Customer Profitability'] - min_prof) / (max_prof - min_prof)) * 100
        else:
            df['Profitability Score'] = 50

    if 'Contract' in df.columns:
        base_risk = np.where(df['Contract'] == 'Month-to-Month', 60, np.where(df['Contract'] == 'One Year', 30, 10))
        tenure_penalty = np.maximum(0, (24 - df['Tenure in Months'])) * 1.5
        df['Churn Risk Score'] = np.clip(base_risk + tenure_penalty + np.random.normal(0, 5, len(df)), 1, 99)

    if 'Churn Risk Score' in df.columns and 'Profitability Score' in df.columns:
        df['Retention Priority Score'] = (df['Churn Risk Score'] * 0.6) + (df['Profitability Score'] * 0.4)

    # ==========================================
    # CUSTOMER VALUE SEGMENTATION (THE ORIGINAL 5)
    # ==========================================
    if 'Net Customer Profitability' in df.columns and 'Churn Risk Score' in df.columns:
        
        prof_top_50 = df['Net Customer Profitability'].quantile(0.50)
        prof_top_70 = df['Net Customer Profitability'].quantile(0.70)
        prof_top_80 = df['Net Customer Profitability'].quantile(0.80)
        
        conditions_seg = [
            # 1. Regrettable Churn: They churned AND were in the top 50% of profit
            (df['Churn Value'] == 1) & (df['Net Customer Profitability'] > prof_top_50),
            
            # 2. VIP: Active, Top 20% Profit, Low Risk
            (df['Churn Value'] == 0) & (df['Net Customer Profitability'] > prof_top_80) & (df['Churn Risk Score'] < 40),
            
            # 3. At risk Premium: Active, Top 30% Profit, High Risk
            (df['Churn Value'] == 0) & (df['Net Customer Profitability'] > prof_top_70) & (df['Churn Risk Score'] >= 60),
            
            # 4. Upsell opportunity: Active, low services, high tenure, low risk
            (df['Churn Value'] == 0) & (df['Service Bundle Count'] <= 2) & (df['Tenure in Months'] > 12) & (df['Churn Risk Score'] < 50)
        ]
        
        choices_seg = ['Regrettable churn', 'VIP', 'At risk Premium', 'Upsell opportunity']
        df['Customer Value Segment'] = np.select(conditions_seg, choices_seg, default='Other')

    # Double Debug Tracker
    total_churners = int(df['Churn Value'].sum())
    regrettable_count = len(df[df['Customer Value Segment'] == 'Regrettable churn'])
    print(f"📊 DEBUG: Total people who churned in raw data: {total_churners}")
    print(f"📊 DEBUG: Found {regrettable_count} Regrettable Churn customers!")

    # ==========================================
    # 5. FINAL CLEANUP & SAVE LOCALLY
    # ==========================================
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    output_filename = 'Model_Ready_Data.csv'
    df.to_csv(output_filename, index=False)
    print(f"✅ Data enriched and cleaned to {len(df.columns)} columns. Preparing to upload...")

    # ==========================================
    # 6. OVERWRITE PLACEHOLDER IN GOOGLE DRIVE
    # ==========================================
    TARGET_FILE_ID = '1m7RHqafoVen_AKSXSKm69lituXBGj2XcD96gR-w63-E'

    try:
        creds_json = os.environ.get('GCP_CREDENTIALS')
        if not creds_json:
            raise ValueError("GCP_CREDENTIALS environment variable not found in GitHub Secrets!")
            
        creds_dict = json.loads(creds_json)
        credentials = service_account.Credentials.from_service_account_info(
            creds_dict, scopes=['https://www.googleapis.com/auth/drive']
        )
        service = build('drive', 'v3', credentials=credentials)
        media = MediaFileUpload(output_filename, mimetype='text/csv', resumable=True)

        updated_file = service.files().update(
            fileId=TARGET_FILE_ID, 
            media_body=media
        ).execute()
        
        print(f"✅ SUCCESS! Overwritten placeholder in Drive. ID: {updated_file.get('id')}")
            
    except Exception as e:
        print(f"❌ Failed to upload to Google Drive: {e}")

if __name__ == '__main__':
    run_pipeline()
