"""
Evaluate the fine-tuned model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
import json

print("="*80)
print("STEP 4: EVALUATING MODEL")
print("="*80)

# Load model
print("\n[1/3] Loading model...")
MODEL_DIR = "models/phishing_detector"
BASE_MODEL = "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, load_in_4bit=True, device_map="auto"
)
model = PeftModel.from_pretrained(base_model, MODEL_DIR)
model.eval()
print("✓ Model loaded")

# Load test data
print("\n[2/3] Loading test data...")
test_data = load_dataset('json', data_files='data/processed/test.json')['train']
print(f"✓ Test samples: {len(test_data)}")

# Predict
print("\n[3/3] Running predictions...")
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                      max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10, 
                                temperature=0.1, do_sample=False)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return "PHISHING" in response.split("Classification:")[-1].upper()

predictions = []
labels = []
for i, example in enumerate(test_data):
    if i % 10 == 0:
        print(f"  Progress: {i}/{len(test_data)}")
    pred = predict(example['text'])
    predictions.append(1 if pred else 0)
    labels.append(example['label'])

# Metrics
print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(classification_report(labels, predictions, 
      target_names=['SAFE', 'PHISHING'], digits=3))

cm = confusion_matrix(labels, predictions)
print(f"\nConfusion Matrix:")
print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")

accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
print(f"\nAccuracy: {accuracy:.3f}")

# Save results
results = {
    'accuracy': float(accuracy),
    'confusion_matrix': cm.tolist(),
}
with open('models/phishing_detector/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✅ EVALUATION COMPLETE!")
print("="*80)