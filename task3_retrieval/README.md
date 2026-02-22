# 🔍 Task 3: Semantic Image Retrieval System

## Overview
This repository contains the implementation of a **Content-Based Image Retrieval (CBIR)** system for medical imaging. Unlike traditional classification, this system allows clinicians to perform "Case-Based Reasoning" by retrieving historical X-rays that are semantically and pathologically similar to a query image or a text description.



## 🛠️ System Components

### 1. Embedding Model: BiomedCLIP
We utilized **BiomedCLIP** (`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`), a state-of-the-art multimodal model. 
* **Medical Latent Space:** Pre-trained on 15 million biomedical image-text pairs from PubMed, ensuring the model understands clinical features like "opacities" and "consolidations" rather than just basic shapes.
* **Feature Vector:** Each image is transformed into a high-dimensional (512-dim) semantic "fingerprint."

### 2. Vector Database: FAISS
For high-speed similarity searching, we implemented **FAISS** (Facebook AI Similarity Search).
* **Cosine Similarity:** By L2-normalizing the embeddings and using an `IndexFlatIP` (Inner Product) index, we ensure that the system retrieves images based on their clinical proximity in the latent space.

---

## 🚀 Features

### 📸 Image-to-Image Search
Input a chest X-ray to find the top-k most similar cases from the database. This is a critical tool for diagnostic verification and finding "look-alike" cases.

### 📝 Text-to-Image Search
Leveraging the multimodal nature of BiomedCLIP, users can search for images using natural language clinical descriptions (e.g., *"severe pneumonia with lung opacities"*).



---

## 📊 Performance Evaluation

The system was rigorously evaluated on the PneumoniaMNIST test set using the **Precision@5** metric. This measures the fraction of the top 5 retrieved images that share the same ground truth label as the query.

| Metric | Result |
| :--- | :--- |
| **Dataset Size** | 624 Test Images |
| **Vector Dimensions** | 512 |
| **Mean Precision@5** | **85.29%** |

**Analysis:** A precision score of **85.29%** demonstrates that the BiomedCLIP latent space is highly effective at clustering similar pathologies together, making it a reliable clinical decision support tool.

---

## 📂 Repository Structure

```text
task3_retrieval_system/
├── src/
│   ├── extract_embeddings.py  # Generates 512-dim vectors for the dataset
│   ├── build_index.py        # Constructs the FAISS vector database
│   └── search_engine.py       # Core retrieval logic (Image & Text modes)
├── data/
│   ├── features.npy           # Saved image embeddings
│   └── medical_index.faiss    # Pre-built vector index
├── requirements.txt           # Dependencies (torch, open_clip, faiss-cpu)
└── task3_retrieval_system.md  # Detailed technical report
```
---
## ⚙️ Installation & Usage
* Install Dependencies:
```
pip install -r requirements.txt
```
* Generate Index:
```
python src/extract_embeddings.py
python src/build_index.py
```
* Run a Search: You can use the MedicalSearchEngine class in src/search_engine.py to perform queries programmatically or via a notebook.