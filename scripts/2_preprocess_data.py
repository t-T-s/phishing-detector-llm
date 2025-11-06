"""
Preprocess phishing data for LLM fine-tuning
"""

import pandas as pd
import json
import re
from sklearn.model_selection import train_test_split
from google.cloud import storage
import os

print("="*80)
print("STEP 2: PREPROCESSING DATA")
print("="*80)

# Load data
print("\n[1/5] Loading data...")
df = pd.read_csv('data/phishing_raw.csv')
print(f"✓ Loaded {len(df)} emails")

# Clean text
print("\n[2/5] Cleaning text...")
def clean_text(text):
    text = str(text)
    text = ' '.join(text.split())
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)
    text = re.sub(r'http\S+|www.\S+', '[URL]', text)
    if len(text) > 1500:
        text = text[:1500] + "..."
    return text

df['text_clean'] = df['text'].apply(clean_text)
df = df[df['text_clean'].str.len() > 50]
print(f"✓ Cleaned: {len(df)} emails remaining")

# Balance dataset
print("\n[3/5] Balancing dataset...")
phishing = df[df['label'] == 1]
safe = df[df['label'] == 0]
min_count = min(len(phishing), len(safe))
phishing = phishing.sample(n=min_count, random_state=42)
safe = safe.sample(n=min_count, random_state=42)
df_balanced = pd.concat([phishing, safe]).sample(frac=1, random_state=42)
print(f"✓ Balanced: {len(df_balanced)} emails")

# Format for training
print("\n[4/5] Formatting for LLM...")
formatted = []
for _, row in df_balanced.iterrows():
    text = f"""Analyze the following email and determine if it is a phishing attempt or legitimate.

Email:
{row['text_clean']}

Classification: {"PHISHING" if row['label']==1 else "SAFE"}"""
    
    formatted.append({
        'text': text,
        'label': int(row['label'])
    })

# Split data
train, temp = train_test_split(formatted, test_size=0.2, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

print(f"✓ Formatted {len(formatted)} emails")
print(f"  Train: {len(train)}")
print(f"  Val: {len(val)}")
print(f"  Test: {len(test)}")

# Save
print("\n[5/5] Saving datasets...")
os.makedirs('data/processed', exist_ok=True)

for name, data in [('train', train), ('val', val), ('test', test)]:
    with open(f'data/processed/{name}.json', 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  ✓ Saved {name}.json")

# Upload to GCS
bucket_name = os.environ.get('BUCKET_NAME')
if bucket_name:
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        for name in ['train', 'val', 'test']:
            blob = bucket.blob(f'data/processed/{name}.json')
            blob.upload_from_filename(f'data/processed/{name}.json')
        print(f"\n✓ Uploaded to GCS: gs://{bucket_name}/data/processed/")
    except Exception as e:
        print(f"⚠ GCS upload failed: {e}")

# Show sample
print("\n" + "="*80)
print("SAMPLE FORMATTED EMAIL:")
print("="*80)
print(train[0]['text'][:500] + "...")
print("="*80)
print("\n✅ PREPROCESSING COMPLETE!")
print("="*80)