# Medical Image Analysis & Retrieval Pipeline

Welcome to the repository for my Postdoctoral Technical Challenge for Alfaisal University. This project demonstrates a comprehensive, end-to-end deep learning pipeline focused on medical image analysis, specifically utilizing the PneumoniaMNIST dataset. 

The repository is structured around three core tasks: image classification, misclassification analysis via super-resolution and UI development, and an advanced content-based image retrieval system.

## 📌 Section 1: Project Summaries

### Task 1: Medical Image Classification
This task focuses on establishing a robust baseline for medical image classification. Using the PneumoniaMNIST dataset, a state-of-the-art vision model (MaxViT) was trained to classify chest X-rays into 'Normal' or 'Pneumonia' categories. The pipeline includes data downloading, preprocessing, model training, and rigorous evaluation using standard classification metrics to ensure high diagnostic reliability.

### Task 2: Misclassification Analysis, Super-Resolution, & UI
Task 2 dives deeper into the model's performance by isolating and analyzing misclassified images from Task 1. To address potential issues caused by the low resolution of MedMNIST images, Super-Resolution techniques (such as SwinIR) are applied to reconstruct fine-grained details. Finally, an interactive, professional User Interface (UI) is developed to visualize these results and facilitate automated Vision-Language Model (VLM) report generation for clinical support.

### Task 3: Content-Based Medical Image Retrieval
The final task implements a scalable image retrieval system. Leveraging the powerful, domain-specific `BiomedCLIP` foundation model, high-dimensional embeddings are extracted from the medical images. These embeddings are indexed using FAISS (Facebook AI Similarity Search) to allow for lightning-fast retrieval of clinically similar X-ray images, demonstrating a critical tool for comparative diagnostics.

---

## 💻 Section 2: Interactive Google Colab Notebooks

You can run and explore the code for each task directly in your browser using the provided Google Colab links below:

### 🚀 Task 1
* [Task 1: Dataset Preparation and Classification Training](https://colab.research.google.com/drive/1M_ggFNRLTXgieaeVHk7WQ4UCv87TLtvu?usp=sharing)

### 🚀 Task 2
* [Task 2.1: ViT Misclassifications Analysis (Without Super-Resolution)](https://colab.research.google.com/drive/1E6XgyDhKwA9nqsp0xtCBAdGGpAkbyL1a?usp=sharing)
* [Task 2.2: Enhancing Misclassifications with Super-Resolution](https://colab.research.google.com/drive/1xGRH-MWNgxhNhHMsoJLguLryF6mu9R7m?usp=sharing)
* [Task 2.3: Interactive Professional UI & Report Generation](https://colab.research.google.com/drive/1xd9MtUtmtykHAfqKeiPFkhqZKDI2xvCy?usp=sharing)

### 🚀 Task 3
* [Task 3: BiomedCLIP & FAISS Retrieval System](https://colab.research.google.com/drive/1UOlPzPXMTsvCiAyhapOnyTn1puQHIibi?usp=sharing)

---

## 🧠 Section 3: Technical Discussion

### Discussion on Task 1: Classification Strategy
The choice of the PneumoniaMNIST dataset presents a unique challenge: achieving high accuracy on highly compressed, low-resolution medical imagery. Utilizing a hybrid architecture like MaxViT allows the model to capture both local features (through convolutions) and global context (through self-attention). The primary challenge here is preventing overfitting on a relatively small dataset while ensuring the model learns generalized features of pneumonia infiltrates rather than dataset artifacts.

### Discussion on Task 2: The Role of Resolution in Diagnostics
Task 2 addresses a fundamental bottleneck in automated medical imaging: the loss of critical morphological details due to downsampling. When a Vision Transformer misclassifies an image, it is often because the distinguishing pathological features have been blurred out. By integrating a Super-Resolution pipeline, we attempt to recover these high-frequency details prior to secondary analysis. Furthermore, wrapping this logic into an interactive UI bridges the gap between raw backend scripting and practical clinical deployment, allowing medical professionals to interact with the AI, view original vs. enhanced images, and generate text-based diagnostic reports seamlessly.

### Discussion on Task 3: Advanced Retrieval with Foundation Models
Standard image retrieval often relies on generic ImageNet-trained models, which fall short when processing specialized medical textures like chest X-rays. Task 3 overcomes this by implementing Microsoft's `BiomedCLIP`, a vision-language foundation model pre-trained specifically on PubMed biomedical literature and image pairs. Because BiomedCLIP inherently understands medical context, the generated embeddings are highly semantically meaningful. By pairing these embeddings with FAISS, the system achieves highly efficient, sub-second nearest-neighbor searches, paving the way for applications like "find similar patient cases" in real-world healthcare settings.
