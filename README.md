# 🧠 End-To-End Deep Learning Project & AI Medical Assistant

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow%202.x-orange)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-brightgreen)](LICENSE)

---

## 🚀 Project Overview

![AI](experiment-notebook/dl-rag.png)
![AI](experiment-notebook/webapp.png)

This project is a **production-ready AI-powered Medical Assistant** that combines:

* 🧠 Deep Learning for **Brain Tumor Detection**
* 🔍 Computer Vision (**VGG16, YOLO, SAM**)
* 📄 Multi-modal **RAG (Retrieval-Augmented Generation)** system
* 🎙️ Voice-based query support using Whisper
* 🤖 LLM-powered medical Q&A system

It enables users to:

* Detect and segment brain tumors from MRI scans
* Ask questions from uploaded PDFs (research papers, reports)
* Perform voice-based medical queries
* Retrieve both **textual and visual answers**

---

## 🏗️ System Architecture

### 🔬 Computer Vision Pipeline

* **VGG16** → Binary Classification (Tumor / No Tumor)
* **YOLO** → Tumor Detection (Bounding Boxes)
* **SAM** → Tumor Segmentation (Pixel-level masks)

### 📚 RAG Pipeline

* PDF ingestion using **PyMuPDF + pdfplumber**
* Text, tables, and images extraction
* Embedding generation using OpenAI
* Storage in **Pinecone Vector Database**
* Context-aware answering using LLM

### 🎤 Voice AI

* Speech-to-text using Whisper
* Query routed through RAG system

---

## ✨ Key Features

### 🖼️ Image Processing

* Automatic resizing to **512x512**
* Supports multiple formats: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`

### 🧠 AI Capabilities

* Tumor classification with confidence score
* Object detection with bounding boxes
* Precise segmentation masks
* Context-aware medical Q&A

### 📄 Multi-Modal Intelligence

* Extracts:

  * Text
  * Tables (converted to Markdown)
  * Images with contextual metadata
* Retrieves relevant visuals during Q&A

### 💻 Frontend

* Clean UI with Tailwind CSS
* Real-time image preview
* Interactive result cards
* Markdown-rendered AI responses

---

## 📂 Project Structure

```
├── app.py
├── main.py
├── store_pincone.py
├── models/
├── final_model/
├── uploads/
├── extracted_images/
├── templates/
├── frontend/
├── src/
│   └── pipeline/
│       └── prediction.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Ahmed2797/End-To-End-Deep-Learning-Project.git
cd End-To-End-Deep-Learning-Project
```

---

### 2️⃣ Create Environment

```bash
conda create -n tumor python=3.10 -y
conda activate tumor
pip install -r requirements.txt
```

---

### 3️⃣ Download Datasets

#### 📌 YOLO Dataset

```text
https://github.com/ultralytics/assets/releases/download/v0.0.0/brain-tumor.zip
```

#### 📌 MRI Classification Dataset

```text
https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection
```

#### 📌 Alternative Dataset

```text
https://drive.google.com/file/d/1OTJ4n6I9uEL9KcOzcQgSmHvUerfwmZWI/view
```

---

### 4️⃣ Environment Variables

Create a `.env` file in root directory:

```env
PINECONE_API_KEY=your_pinecone_api_key
OPENAI_API_KEY=your_openai_api_key
```

---

## ▶️ Running the Application

```bash
python store_pincone.py
python main.py
python app.py
```

Or directly:

```bash
uvicorn app:app --reload
```

---

## 🔌 API Endpoints

### 📄 RAG System

* `POST /upload` → Upload PDF
* `GET /ask?q=...` → Ask questions

### 🎤 Voice Query

* `POST /voice-query`

### 🧠 Model Inference

* `POST /predict_vgg` → Classification
* `POST /detect` → Object Detection
* `POST /segment` → Segmentation

### 🟢 Health Check

* `GET /api`

---

## 🔄 Workflow

### 🧠 Image Processing

1. Upload MRI image
2. Choose:

   * VGG → classification
   * YOLO → detection
   * SAM → segmentation
3. View results instantly

### 📄 PDF RAG

1. Upload PDF
2. System extracts:

   * Text
   * Tables
   * Images
3. Ask questions
4. Get:

   * Contextual answer
   * Relevant images

### 🎤 Voice Query

1. Upload audio
2. Convert speech → text
3. Query RAG system
4. Return answer

---

## 📊 Tech Stack

* **Backend:** FastAPI
* **Deep Learning:** TensorFlow, PyTorch
* **CV Models:** YOLO, SAM, VGG16
* **LLM:** OpenAI API
* **Vector DB:** Pinecone
* **Speech:** Whisper
* **PDF Processing:** PyMuPDF, pdfplumber
* **Frontend:** HTML, Tailwind CSS

---

## 📈 Future Improvements

* Real-time webcam tumor detection
* Multilingual medical assistant
* Advanced report generation
* Model optimization for edge devices
* Deployment on AWS/GCP

---

## 📬 Contact

**Author:** Ahmed Tanvir
🔗 GitHub: https://github.com/Ahmed2797

**Interests:**

* Machine Learning Engineering
* Medical AI
* Computer Vision

---

## 📜 License

This project is licensed under the **MIT License**.

---

## ⭐ Final Note

This project demonstrates a **complete end-to-end AI system** combining:

* Deep Learning
* Multi-modal RAG
* Real-time inference
* Scalable backend architecture

Perfect for:

* 💼 Portfolio Projects
* 🎯 ML Interviews
* 🏥 Medical AI Applications

---
