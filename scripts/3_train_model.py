"""
Fine-tune LLM for phishing detection
"""

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from google.cloud import storage
import json, os
from datetime import datetime

print("="*80)
print("STEP 3: FINE-TUNING LLM")
print("="*80)
print(f"Start: {datetime.now()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("="*80)

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
OUTPUT_DIR = "models/phishing_detector"

# Load dataset
print("\n[1/6] Loading dataset...")
dataset = load_dataset('json', data_files={
    'train': 'data/processed/train.json',
    'validation': 'data/processed/val.json',
})
print(f"✓ Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}")

# Load model
print(f"\n[2/6] Loading {MODEL_NAME}...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)
print(f"✓ Model loaded: {model.num_parameters()/1e9:.1f}B parameters")

# Configure LoRA
print("\n[3/6] Configuring LoRA...")
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Tokenize
print("\n[4/6] Tokenizing...")
def tokenize(examples):
    tok = tokenizer(examples['text'], truncation=True, 
                   max_length=512, padding="max_length")
    tok["labels"] = tok["input_ids"].copy()
    return tok

tokenized = dataset.map(tokenize, batched=True, 
                       remove_columns=['text', 'label'])
print("✓ Tokenization complete")

# Training config
print("\n[5/6] Configuring training...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# Train!
print("\n[6/6] Training...")
print("="*80)
torch.cuda.empty_cache()
result = trainer.train()
print("="*80)
print(f"✓ Training complete! Loss: {result.training_loss:.4f}")

# Save
os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✓ Model saved: {OUTPUT_DIR}")

# Upload to GCS
bucket_name = os.environ.get('BUCKET_NAME')
if bucket_name:
    print("\nUploading to GCS...")
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        for root, dirs, files in os.walk(OUTPUT_DIR):
            for file in files:
                local = os.path.join(root, file)
                remote = f"models/{os.path.relpath(local, OUTPUT_DIR)}"
                bucket.blob(remote).upload_from_filename(local)
        print(f"✓ Uploaded to gs://{bucket_name}/models/")
    except Exception as e:
        print(f"⚠ Upload failed: {e}")

print("\n"+"="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)