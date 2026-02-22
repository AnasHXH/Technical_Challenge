# 🏥 Task 2: Medical Report Generation using Visual Language Models (VLMs)

## Overview
This repository contains the implementation for Task 2 of the Postdoctoral Technical Challenge. It demonstrates an advanced multimodal AI pipeline for automated medical report generation from chest X-rays. The project bridges the gap between standard classification and cutting-edge generative AI, evaluating how resolution constraints and super-resolution techniques affect the clinical reasoning of Visual Language Models.

## Models Used

### 1. MedGemma 1.5 4B-IT (Report Generation)
* **Description:** We utilize `google/medgemma-1.5-4b-it`, an open-weights multimodal model built on Google's Gemma 3 architecture. It is specifically instruction-tuned for medical text reasoning and image comprehension.
* **Specifications:** With 4 Billion parameters, it provides an excellent balance between high-fidelity clinical reasoning and computational efficiency. Furthermore, it inherits robust multilingual capabilities, allowing for report generation and interactive prompting in multiple languages, including Arabic (العربية).

**multiple languages:**
![multiple languages](https://github.com/AnasHXH/Task_2/blob/main/app_2.png)

---
### 2. MaxViT (Classification Baseline)
* **Description:** Carried over from Task 1, MaxViT serves as our baseline.
* **Role:** Instead of testing random images, we specifically extract the edge cases that MaxViT misclassified. By feeding these challenging images into the VLM, we perform a strict qualitative analysis comparing standard classification against multimodal AI reasoning.
** Download the weight: [Google Drive Link](https://drive.google.com/file/d/1DFWhF_euBfmUaS-2XWgMnkZgsfxvvJc6/view?usp=sharing)

---
### 3. Swin2SR / SwinIR (AI Super-Resolution)
* **Description:** The Swin Transformer V2 for Image Super-Resolution (`caidas/swin2SR-classical-sr-x4-64`).
* **Role:** MedGemma requires high-resolution inputs (224x224) to analyze anatomical structures. Because the MedMNIST dataset contains highly compressed 28x28 images, we implement an experimental Swin2SR pipeline to upscale the images. This allows us to critically evaluate the trade-offs between mathematical upsampling (Lanczos) and the risk of AI-generated "hallucinated" medical textures.

**AI Super-Resolution:**
![AI Super-Resolution](https://github.com/AnasHXH/Task_2/blob/main/SwinIr.png)

---
## ⚙️ Installation & Setup
To replicate this environment, ensure you have Python 3.10+ installed, then install the required dependencies:

```bash
pip install -r requirements.txt
```
* Note on Hugging Face Authentication:
  To download the MedGemma model weights, you must have a Hugging Face account and accept the Health AI Developer Foundations terms of use. Make sure to authenticate your environment using your access token before running the scripts:
  ```bash
  huggingface-cli login
  ```
---

## 📂 Notebooks Overview & File Descriptions

This task is divided into three comprehensive Jupyter notebooks, each serving a specific role in evaluating the VLM pipeline:

### 1. `Task_2_Vit_Misclassifications.ipynb`
* **Purpose:** This notebook establishes the baseline performance of the MedGemma VLM. 
* **What it does:** It specifically targets the 10 edge-case images that the MaxViT CNN misclassified in Task 1. The highly compressed $28 \times 28$ MedMNIST images are mathematically upsampled to $224 \times 224$ using standard Lanczos resampling. It then runs an automated evaluation loop to compare MedGemma's text-based diagnosis against both the Ground Truth and the CNN's failed predictions.
* **Key Insight:** Demonstrates how VLMs trained on high-resolution clinical X-rays heavily default to predicting "Normal" when presented with blurry, upscaled inputs lacking fine-grained pathological textures.

### 2. `Task_2_Super_Resolution_Vit_Misclassifications.ipynb`
* **Purpose:** An experimental pipeline investigating whether AI-driven Super-Resolution can bridge the domain gap between low-resolution datasets and high-resolution VLMs.
* **What it does:** Replaces the standard mathematical upscaling with `Swin2SR` (Swin Transformer V2 for Image Super-Resolution). The $28 \times 28$ images are first enhanced by the AI to $112 \times 112$ before final sizing. It includes side-by-side visual comparisons of the original vs. super-resolution images before passing them to MedGemma.
* **Key Insight:** Evaluates the critical clinical trade-off between improving image clarity for the VLM and the risk of the Super-Resolution model "hallucinating" textures that mimic pneumonia infiltrates.

### 3. `Task_2_Interactive_Professional_UI.ipynb`
* **Purpose:** A deployment-ready, interactive web interface for real-time clinical prompting.
* **What it does:** Wraps the MedGemma pipeline in a professional Gradio UI. It allows users to manually select upscaled images, adjust generation parameters (Temperature, Max Tokens, Top-P), and test different prompt engineering strategies (e.g., Zero-Shot vs. Structured Role-Prompting). 
* **Key Feature:** Leverages MedGemma's multilingual capabilities to generate diagnostic reports in multiple languages, including English and Arabic (العربية).
---
## 🔗 Interactive Google Colab Notebooks

To facilitate easy testing and demonstration without requiring local environment setup, the complete pipeline for Task 2 is available as interactive Google Colab notebooks. You can view, run, and interact with the code using the links below:

* **[Task_2: Vit_Misclassifications.ipynb](https://colab.research.google.com/drive/1E6XgyDhKwA9nqsp0xtCBAdGGpAkbyL1a?usp=sharing)**
  * *Description:* Evaluates the baseline MedGemma VLM performance on the exact 10 images that the MaxViT CNN misclassified, using standard Lanczos upsampling.

* **[Task_2: Super_Resolution_Vit_Misclassifications.ipynb](https://colab.research.google.com/drive/1xGRH-MWNgxhNhHMsoJLguLryF6mu9R7m?usp=sharing)**
  * *Description:* Demonstrates the experimental AI super-resolution pipeline using Swin2SR to upscale the low-resolution MedMNIST images before feeding them to the VLM.

* **[Task_2: Interactive_Professional_UI.ipynb](https://colab.research.google.com/drive/1xd9MtUtmtykHAfqKeiPFkhqZKDI2xvCy?usp=sharing)**
  * *Description:* A deployment-ready Gradio web interface that allows users to interactively test the MedGemma model with different prompting strategies and multiple languages.

---

## 📈 Results & Evaluation

This section evaluates MedGemma's ability to act as a "second-opinion" diagnostic tool. Specifically, we tested the VLM exclusively on the 10 edge-case images that the **MaxViT CNN misclassified** in Task 1. 

### Baseline VLM Performance (Without Super-Resolution)
Initially, the low-resolution ($28 \times 28$) MedMNIST images were mathematically upsampled to $224 \times 224$ using standard Lanczos resampling to meet MedGemma's input requirements. 

* **Result:** The VLM achieved a **30.0% accuracy** on these CNN failure cases.
* **Analysis:** Because MedGemma was trained on high-resolution adult X-rays, the blurry, mathematically stretched images lacked the fine-grained pathological textures (like consolidations or infiltrates) the model expects. Consequently, the VLM heavily defaulted to predicting "Normal," failing to confidently identify pneumonia.

![Before SR Results](https://raw.githubusercontent.com/AnasHXH/Task_2/main/before_SR.png)

### Enhanced VLM Performance (With Swin2SR)
To bridge the resolution gap, we replaced the Lanczos resampling with an AI-driven Super-Resolution pipeline using **Swin2SR** (Swin Transformer V2). The images were AI-upscaled to $112 \times 112$ before final sizing, recovering and enhancing missing structural details.

* **Result:** The VLM accuracy doubled to **60.0%** on the exact same CNN failure cases.
* **Analysis:** The super-resolution model successfully sharpened anatomical boundaries and highlighted latent opacities. With clearer visual inputs, MedGemma's clinical reasoning capabilities were unlocked, allowing it to correctly diagnose the majority of the images. 

![After SR Results](https://raw.githubusercontent.com/AnasHXH/Task_2/main/after_SR.png)

### Conclusion: Fixing CNN Misclassifications
The most significant finding of this experiment is the complementary strength of Multimodal AI. While the standard MaxViT CNN completely failed on these 10 specific images, **the MedGemma VLM (when paired with Swin2SR) successfully corrected 6 out of 10 of those misclassifications.** This demonstrates that integrating a Visual Language Model into the diagnostic pipeline can effectively catch and correct the errors of traditional classification networks, paving the way for more robust and resilient automated healthcare systems.





