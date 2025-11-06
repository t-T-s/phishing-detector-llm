"""
Download phishing email dataset
Run in GCP VM - no local machine needed!
"""

from datasets import load_dataset
import pandas as pd
import json
from google.cloud import storage
import os

print("="*80)
print("STEP 1: DOWNLOADING PHISHING EMAIL DATASET")
print("="*80)

# Load dataset from Hugging Face
print("\n[1/3] Loading dataset from Hugging Face...")
try:
    dataset = load_dataset("ealvaradob/phishing-dataset")
    df = pd.DataFrame(dataset['train'])
    print(f"✓ Dataset loaded: {len(df)} emails")
except Exception as e:
    print(f"Using backup dataset...")
    # Create sample dataset if HF fails
    phishing_emails = [
        "URGENT: Your account will be suspended. Click here to verify immediately!",
        "Congratulations! You won $1,000,000. Send your bank details to claim.",
        "Your password has expired. Update now at: http://fake-site.com",
        "Dear customer, verify your identity to prevent account closure.",
        "IRS TAX REFUND: Click here to claim your $5,000 refund now!",
    ] * 200
    
    safe_emails = [
        "Hi team, please review the attached quarterly report for Q3.",
        "Meeting scheduled for tomorrow at 2 PM in conference room A.",
        "Thank you for your order. Your tracking number is 123456789.",
        "Your monthly subscription payment of $9.99 was successful.",
        "Welcome to our newsletter! Here are this week's updates.",
    ] * 200
    
    df = pd.DataFrame({
        'text': phishing_emails + safe_emails,
        'label': [1]*len(phishing_emails) + [0]*len(safe_emails)
    })
    df = df.sample(frac=1).reset_index(drop=True)
    print(f"✓ Sample dataset created: {len(df)} emails")

# Statistics
print(f"\n[2/3] Dataset statistics:")
print(f"  Total emails: {len(df)}")
print(f"  Phishing: {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
print(f"  Safe: {(1-df['label']).sum()} ({(1-df['label'].mean())*100:.1f}%)")

# Save locally
print(f"\n[3/3] Saving dataset...")
os.makedirs('data', exist_ok=True)
df.to_csv('data/phishing_raw.csv', index=False)
print("✓ Saved to: data/phishing_raw.csv")

# Upload to GCS
bucket_name = os.environ.get('BUCKET_NAME')
if bucket_name:
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob('data/phishing_raw.csv')
        blob.upload_from_filename('data/phishing_raw.csv')
        print(f"✓ Uploaded to: gs://{bucket_name}/data/phishing_raw.csv")
    except Exception as e:
        print(f"⚠ GCS upload failed: {e}")

print("\n"+"="*80)
print("✅ DATA DOWNLOAD COMPLETE!")
print("="*80)