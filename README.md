# 🛡️ Phishing Email Detection on GCP

### *A Complete Cloud-Only Machine Learning Project — No Local Setup Required!*

This project demonstrates a **fully cloud-based pipeline** for detecting phishing emails using **LLM fine-tuning on Google Cloud Platform (GCP)**. Everything — from data download to API deployment — runs entirely in the cloud.

---

## 🚀 Features

* 100% Cloud-Based (no local machine needed)
* End-to-end ML pipeline (data → model → API)
* LoRA fine-tuning of Mistral-7B model
* GPU-enabled VM setup with one command
* FastAPI REST endpoint + web demo interface
* Integrated with Google Cloud Storage (GCS)

---

## 🧰 Tools & Technologies

* **GCP**: Compute Engine, Cloud Storage, Cloud Shell
* **Python**: Transformers, PEFT, Datasets, FastAPI, Uvicorn
* **Model**: Mistral-7B-v0.1 (fine-tuned via LoRA)

---

## 📋 Quick Start

### 1️⃣ Setup (Cloud Shell)

```bash
# Create project and bucket
export PROJECT_ID="phishing-detection-$(date +%s)"
gcloud projects create $PROJECT_ID
gcloud config set project $PROJECT_ID
gsutil mb -l us-central1 gs://phishing-${PROJECT_ID}
```

### 2️⃣ Create GPU VM

```bash
gcloud compute instances create phishing-vm \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=200GB
```

### 3️⃣ SSH and Run Pipeline

```bash
gcloud compute ssh phishing-vm --zone=us-central1-a
cd ~/phishing-project
python scripts/1_download_data.py
python scripts/2_preprocess_data.py
python scripts/3_train_model.py
python scripts/4_evaluate_model.py
```

### 4️⃣ Deploy & Test API

```bash
nohup python api/app.py > logs/api.log 2>&1 &
python scripts/5_test_api.py
```

---

## 🌐 Browser Access

| Service  | URL Example                           |
| -------- | ------------------------------------- |
| API Docs | `http://<EXTERNAL_IP>:8000/docs`      |
| Web Demo | `http://<EXTERNAL_IP>:8080/demo.html` |

---

## 📁 Project Structure

```
phishing-project/
├── data/processed/         # Train, val, test data
├── models/phishing_detector/  # Fine-tuned LLM + results
├── scripts/                # Data, training, evaluation, API tests
├── api/                    # FastAPI app + HTML demo
└── logs/                   # API logs
```

---

## 💸 Cost & Duration

| Resource               | Cost/hour  | Notes                  |
| ---------------------- | ---------- | ---------------------- |
| n1-standard-4 + T4 GPU | ~$0.54     | Main VM                |
| Storage (200GB)        | ~$0.10/day | GCS bucket             |
| Total                  | ~$2–3      | Full project (3–4 hrs) |

---

## ✅ Deliverables

* Fine-tuned LLM for phishing detection
* REST API accessible via browser
* Web-based email testing demo
* Full reproducibility using only cloud resources

---

## 💡 Ideal For

* Cloud ML project demonstrations
* End-to-end deployment tutorials
* Interview or portfolio showcases

---

**Author:** *Your Name*
**License:** MIT

> “Train, deploy, and demo — entirely in the cloud ☁️.”

---