# Task 3: Semantic Image Retrieval System

## 1. Embedding Model Selection and Justification
For this semantic retrieval system, we implemented **BiomedCLIP** (`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`). 

**Justification:**
Traditional vision models (like ResNet or standard CLIP) are trained on general natural images and lack the "medical vocabulary" to differentiate subtle pathological features in chest X-rays. BiomedCLIP is pretrained on **15 million biomedical image-text pairs** using a PubMedBERT text encoder. This ensures that the latent space is structured according to clinical semantics, making it ideal for identifying diagnostic similarities rather than just visual ones.

## 2. Vector Database Implementation
The system utilizes **FAISS (Facebook AI Similarity Search)** for efficient nearest-neighbor retrieval.

* **Feature Extraction:** All 624 images in the PneumoniaMNIST test set were passed through the BiomedCLIP vision encoder to generate 512-dimensional embeddings.
* **Normalization:** All vectors were L2-normalized to ensure that Inner Product (IP) calculations in FAISS are mathematically equivalent to **Cosine Similarity**.
* **Indexing:** We utilized `faiss.IndexFlatIP`, providing an exhaustive and precise search across the semantic space.



## 3. Retrieval Architecture and Usage
The implementation supports two distinct clinical interfaces:

### 3.1 Image-to-Image (Case-Based Reasoning)
Allows a clinician to input a new X-ray to find the most similar historical cases.
* **Input:** Query Image (PIL/Tensor).
* **Process:** Vision encoding -> FAISS Vector Search -> Metadata Retrieval.

### 3.2 Text-to-Image (Semantic Search)
Allows users to search the image database using natural language descriptions (e.g., *"severe pneumonia with lung opacities"*).
* **Input:** Natural Language Query.
* **Process:** Text encoding via PubMedBERT -> Vector space projection -> FAISS lookup.

## 4. Quantitative Evaluation
The system was evaluated using **Precision@k** where $k=5$. For every image in the test set, the system retrieved the top 5 nearest neighbors and checked for label consistency.

| Metric | Result |
| :--- | :--- |
| **Total Test Queries** | 624 Images |
| **Mean Precision@5** | **85.29%** |

This high precision score indicates that the BiomedCLIP embeddings successfully cluster clinically similar cases (Pneumonia vs. Normal) with high density in the latent space.



## 5. Qualitative Analysis & Failure Cases
The visualization of top-$k$ results demonstrates that the system effectively ignores common radiographic noise and focuses on lung density and anatomical boundaries.

**Failure Case Analysis:**
In the cases where the system retrieved an incorrect label (False Positives), the following was observed:
1.  **Resolution Constraints:** Since the dataset uses $28 \times 28$ images, the model occasionally relies on structural similarities (ribcage shape, patient size) rather than subtle pneumonia infiltrates that are lost during downsampling.
2.  **Visual Overlap:** Early-stage pneumonia or mild atelectasis can visually mimic "Normal" scans in a compressed latent space, leading the retrieval engine to group them together.

## 6. Conclusion
The retrieval system acts as a robust **Clinical Decision Support System (CDSS)**. By correctly identifying similar cases 85.2% of the time, it provides clinicians with evidence-based historical references that go beyond a simple "Pneumonia/Normal" classification.