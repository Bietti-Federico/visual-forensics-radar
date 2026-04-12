<h1 align="center">🛡️ Visual Forensics Radar</h1>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+">
   <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

**Visual Forensics Radar** is a sophisticated, multi-layered AI orchestration system designed to detect digital forgeries, manual image manipulations, and AI-generated synthetic media. By combining mathematical error analysis, semantic latent space detection, and visual-language reasoning, it provides a comprehensive "forensic report" for any suspicious image.

![visual-forensics-demo](images/image1.png)  

---
## 🤗 Hugging Face Demo
You can try the live version of the project here:

**[Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/kadircancelik/Fake-Detector)**

>[!IMPORTANT]
>* **Technical Performance:** The demo runs on Hugging Face's Free CPU Tier. Since the system operates without a GPU:
>* **Cold Start:** The initial loading of the AI engines may take 2-3 minutes.
>* **Inference Speed:** Processing an image through all three layers (especially the VLM Layer) can take between 90 to 180 seconds.
>* **Memory Optimization:** The Qwen2-VL model is forced to load in float32 on CPU memory.
>* **Please be patient while the "Detective" analyzes the pixels and logic!**

---

## 📂 Repository Structure

├── backend/  
│   ├── ela_detector.py      # Error Level Analysis Engine  
│   ├── clip_detector.py     # Semantic Deepfake Detector  
│   └── vlm_explainer.py     # Qwen2-VL Reasoning Logic  
├── frontend/  
│   └── app.py               # Streamlit Dashboard  
├── .streamlit/  
│   └── config.toml          # Streamlit Configurations   
├── main.py                  # FastAPI Entrypoint  
├── Dockerfile               # Containerization blueprint  
├── start.sh                 # Startup script for Docker  
└── requirements.txt         # Dependencies  

---

## 🏗️ System Architecture

The project employs a decoupled architecture consisting of a **FastAPI** backend and a **Streamlit** frontend, orchestrated via **Docker**. It utilizes an "Ensemble of Experts" approach across three distinct forensic layers:

### 🔍 Layer 1: Pixel-Level Analysis (ELA)
* **Engine:** `ElaDetector`
* **Logic:** Uses **Error Level Analysis** to detect compression inconsistencies.
* **Target:** Identifies manual "splicing" or "Photoshop" edits where specific regions have different compression levels than the rest of the image.

### 🧠 Layer 2: Semantic Analysis (CLIP)
* **Engine:** `ClipAuthenticator`
* **Logic:** Leverages OpenAI's **CLIP** (Contrastive Language-Image Pre-training) for Zero-Shot classification.
* **Target:** Detects the "latent DNA" of generative models (like Midjourney or DALL-E) by comparing image embeddings against known synthetic vs. real descriptors.

### 🕵️ Layer 3: Logical Reasoning (VLM)
* **Engine:** `VlmExplainer` 
* **Logic:** Utilizes **Qwen2-VL-2B**, a state-of-the- visual-language model.
* **Target:** Acts as a "Digital Detective" that identifies physical impossibilities (e.g., asymmetrical shadows, distorted limbs, or unnatural lighting) and explains them in natural language.

---

## 🛠️ Tech Stack

* **API:** FastAPI
* **Frontend:** Streamlit
* **ML Frameworks:** PyTorch, Transformers (Hugging Face)
* **Models:** `openai/clip-vit-base-patch32`, `Qwen/Qwen2-VL-2B-Instruct` 
* **Deployment:** Docker

---

## 🚀 Local Installation

Ensure you have **Docker** installed on your machine.

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/KadirCanCelik/visual-forensics-radar.git](https://github.com/KadirCanCelik/visual-forensics-radar.git)
   cd visual-forensics-radar
2. **Build the Docker Image:**
   ```bash
   docker build -t visual-forensics-radar .
3. **Run the Container:**
   ```bash
   docker run -p 8501:8501 -p 8000:8000 visual-forensics-radar
4. **Access the Dashboard:**
   Open your browser and navigate to `http://localhost:8501`.

