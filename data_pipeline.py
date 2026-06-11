import pandas as pd
import numpy as np

def run_pipeline():
    print("🚀 Starting DataPipeline...")

    # 1. DOWNLOAD THE RAW DATA FROM GOOGLE SHEETS
    SHEET_ID = '1snki1i6rpKpVjOpk22WbUd6brh3ZSl71p6Hy-uh5mPE'
    SHEET_NAME = 'Sheet1' # Change if your tab is named differently
    
    url = f'https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}'
    
    try:
        df = pd.read_csv(url)
        print(f"✅ Successfully downloaded {len(df)} rows from Google Sheets.")
    except Exception as e:
        print(f"❌ Failed to download data: {e}")
        return

    # 2. DATA CLEANING
    print("🧹 Cleaning data...")
    
    # Fill missing Internet Types with 'None'
    if 'Internet Type' in df.columns:
        df['Internet Type'] = df['Internet Type'].replace(r'^\s*$', np.nan, regex=True)
        df['Internet Type'] = df['Internet Type'].fillna('None')

    # Fix Excel date formatting errors in Tenure Group (e.g., '7-Dec' becomes 'Months 7-12')
    if 'Tenure Group' in df.columns:
        df['Tenure Group'] = df['Tenure Group'].replace('7-Dec', 'Months 7-12')

    # Ensure financial columns are numeric, filling bad data with 0
    financial_cols = ['Monthly Charge', 'Total Charges', 'Total Refunds']
    for col in financial_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 3. FEATURE ENGINEERING
    print("⚙️ Engineering new features...")

    # A. Interaction Velocity (Calls per month of tenure)
    if 'Interaction Frequency (Annual)' in df.columns and 'Tenure in Months' in df.columns:
        df['Interaction Velocity'] = df['Interaction Frequency (Annual)'] / np.maximum(1, df['Tenure in Months'] / 12)

    # B. Estimated Cost-to-Serve (Assuming each interaction costs $15 to handle)
    if 'Interaction Frequency (Annual)' in df.columns:
        df['Estimated Cost-to-Serve'] = df['Interaction Frequency (Annual)'] * 15.0

    # C. Net Customer Profitability
    if all(c in df.columns for c in ['Total Revenue', 'Estimated Cost-to-Serve', 'Total Refunds']):
        df['Net Customer Profitability'] = df['Total Revenue'] - df['Estimated Cost-to-Serve'] - df['Total Refunds']

    # D. Contract Risk Flag (1 if Month-to-Month AND under 1 year tenure)
    if 'Contract' in df.columns and 'Tenure in Months' in df.columns:
        df['Contract Risk Flag'] = np.where(
            (df['Contract'] == 'Month-to-Month') & (df['Tenure in Months'] <= 12), 1, 0
        )

    # E. Service Bundle Count (Tallying how many services they subscribe to)
    service_cols = ['Phone Service', 'Multiple Lines', 'Internet Service', 'Online Security', 
                    'Online Backup', 'Device Protection Plan', 'Premium Tech Support', 
                    'Streaming TV', 'Streaming Movies', 'Streaming Music']
    
    # Only calculate if the columns exist in the raw data
    available_services = [col for col in service_cols if col in df.columns]
    if available_services:
        # Assuming 'Yes'/1 means they have the service
        df['Service Bundle Count'] = df[available_services].apply(
            lambda x: x.map({'Yes': 1, 'No': 0, 1: 1, 0: 0}).fillna(0).sum(), axis=1
        )

    # F. Refund Rate (%)
    if 'Total Refunds' in df.columns and 'Total Revenue' in df.columns:
        df['Refund Rate (%)'] = (df['Total Refunds'] / np.maximum(1, df['Total Revenue'])) * 100
        df['Refund Rate (%)'] = df['Refund Rate (%)'].round(2)


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
        
        # Connect to Google Drive
        service = build('drive', 'v3', credentials=credentials)

        # Prepare the file to upload
        media = MediaFileUpload(output_filename, mimetype='text/csv', resumable=True)

        # Directly overwrite the exact file
        updated_file = service.files().update(
            fileId=TARGET_FILE_ID, 
            media_body=media
        ).execute()
        
        print(f"✅ Successfully overwritten the specific file in Google Drive! File ID: {updated_file.get('id')}")

    except Exception as e:
        print(f"❌ Failed to upload to Google Drive: {e}")

if __name__ == '__main__':
    run_pipeline()
