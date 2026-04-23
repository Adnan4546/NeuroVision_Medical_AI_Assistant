# NeuroVision_Medical_AI_Assistant
AI-powered Brain Tumor Detection system using CNN, Grad-CAM (XAI), and RAG-based Medical Chatbot with LLM integration for MRI analysis and medical Q&amp;A.


# 🧠 Brain Tumor AI Platform (XAI + RAG + LLM)

## 🚀 Overview
AI-powered system for brain tumor detection, explanation, and medical question answering.

## 🔥 Features

### 🧠 MRI Tumor Detection
- CNN-based classification
- Classes:
  - Glioma
  - Meningioma
  - Pituitary
  - No Tumor
- Confidence score

### 🔍 Explainable AI (Grad-CAM)
- Visual heatmap of tumor region
- Improves model transparency

### 🤖 Medical Chatbot (RAG + LLM)
- Uses knowledge base (`tumor_knowledge.txt`)
- Conversational memory
- Powered by LLM via API

### 📄 PDF Analyzer
- Upload medical PDFs
- Ask questions from document
- Uses embeddings + vector search

---

## 🏗️ Architecture

MRI → CNN(VGG16) → Prediction
↓
Grad-CAM (XAI)
↓
User Query → RAG → LLM → Answer
↓
Streamlit UI


---

##  Tech Stack

- TensorFlow / Keras  
- CNN (VGG16)  
- Sentence Transformers  
- FAISS  
- Streamlit  
- LLM API (Groq)

---

## ▶️ Run Project

```bash
git clone https://github.com/Adnan4546/NeuroVision_Medical_AI_Assistant
cd brain-tumor-ai
pip install -r requirements.txt
streamlit run app.py
