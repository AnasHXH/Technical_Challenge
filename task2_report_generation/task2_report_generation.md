# Task 2: Medical Report Generation using Visual Language Model

## 1. Model Selection Justification
For this task, we selected **MedGemma 1.5 4B-IT** (`google/medgemma-1.5-4b-it`). The decision to integrate this specific Visual Language Model (VLM) is based on several critical factors:
* **Domain-Specific Tuning:** Unlike general-purpose VLMs (such as standard LLaVA), MedGemma is explicitly instruction-tuned on diverse medical datasets, enabling it to utilize accurate clinical terminology and anatomical reasoning.
* **Architectural Efficiency:** Built on the Gemma 3 architecture with 4 billion parameters, it strikes an optimal balance. It is small enough to run efficiently on consumer-grade hardware (or the free tier of Google Colab) while retaining high-fidelity reasoning capabilities.
* **Multilingual Capacity:** The model natively supports multiple languages. This allowed us to build an interactive UI that generates reports in both English and Arabic, highly relevant for localized deployment in Saudi healthcare systems.

![Multilingual Generation in Arabic](https://raw.githubusercontent.com/AnasHXH/Task_2/main/app_2.png)

## 2. Prompting Strategies Tested and Their Effectiveness
During the implementation of the interactive UI and the automated batch evaluation, several prompting strategies were tested:

1. **Zero-Shot / Open-Ended Prompt:** *(e.g., "Describe this chest X-ray.")*
   * *Effectiveness:* Low. The model produced overly brief or generalized statements, often skipping specific anatomical structures unless they had glaring abnormalities.
2. **Structured Role-Prompting:** *(e.g., "Examine this chest X-ray. 1. Describe the lungs. 2. Describe the heart. 3. Conclude if pneumonia opacities are present.")*
   * *Effectiveness:* High. Forcing the model to sequentially evaluate the lungs, then the heart, and finally synthesize a conclusion drastically reduced hallucinations. It ensured a standardized output format that is much easier to parse programmatically for evaluation. This strategy was utilized for the final automated batch processing.

![Interactive Professional UI](https://raw.githubusercontent.com/AnasHXH/Task_2/main/app.png)

## 3. Sample Generated Reports
Below are representative samples of the VLM's output using the structured prompt on the 28x28 MedMNIST images (upscaled via Lanczos).

**Sample 1: True Negative (Correctly identified as Normal)**
* **Index:** 61
* **Ground Truth:** Normal
* **ViT (MaxViT) Prediction:** Pneumonia (Failed)
* **VLM Generated Text:**
  > "1. **Lungs:** The lungs appear relatively clear, with no obvious consolidation or significant infiltrates. The lung markings are visible, suggesting adequate lung volume.
  > 2. **Heart:** The heart size appears within normal limits. The cardiomediastinal silhouette is unremarkable.
  > 3. **Pneumonia Opacities:** There are no clear opacities or consolidation patterns suggestive of pneumonia in the lungs.
  > **Conclusion:** The chest X-ray shows no evidence of pneumonia."

**Sample 2: False Negative (Failed to identify Pneumonia)**
* **Index:** 22
* **Ground Truth:** Pneumonia
* **ViT (MaxViT) Prediction:** Normal (Failed)
* **VLM Generated Text:**
  > "1. **Lungs:** The lungs appear relatively clear, with no obvious consolidation or significant infiltrates. However, the image quality is poor, making a definitive assessment difficult. There is some haziness in the lower lung fields...
  > 2. **Heart:** The heart size appears within normal limits based on the limited view.
  > 3. **Pneumonia Opacities:** Given the poor image quality, it is difficult to definitively rule out pneumonia. There are no clear opacities that would strongly suggest pneumonia."

## 4. Qualitative Analysis: VLM vs. Ground Truth vs. ViT
To rigorously test the VLM, we evaluated it exclusively on a curated subset of 10 edge-case images that the **MaxViT ViT misclassified** in Task 1.

* **Baseline VLM Performance (Lanczos Upscaling):** The VLM achieved a **30.0% accuracy** on these ViT failure cases. The qualitative outputs (like Sample 2 above) reveal *why* it failed: the model repeatedly noted that "the image quality is poor." Because the input was a blurry 28x28 image mathematically stretched to 224x224, the fine-grained infiltrates indicative of pneumonia were destroyed. Lacking clear evidence, the model safely defaulted to predicting "Normal."

![VLM Evaluation Before Super-Resolution](https://raw.githubusercontent.com/AnasHXH/Task_2/main/before_SR.png)

* **Enhanced VLM Performance (Swin2SR Upscaling):** When the Lanczos resampling was replaced with an AI Super-Resolution model (Swin Transformer V2), the VLM's accuracy on these exact same failure cases jumped to **60.0%**. The SR model sharpened anatomical boundaries, unlocking the VLM's clinical reasoning and allowing it to successfully correct 6 out of the 10 original ViT misclassifications.

![VLM Evaluation After Super-Resolution](https://raw.githubusercontent.com/AnasHXH/Task_2/main/after_SR.png)

## 5. Strengths and Limitations
**Strengths:**
* **Second-Opinion Capability:** The VLM proved it can successfully override and correct a traditional ViT when provided with adequate visual fidelity.
* **Explainability:** Unlike the ViT, which outputs a black-box probability score, the VLM provides a natural language rationale for its diagnosis, noting specifically when image quality limits its confidence.

**Limitations:**
* **Domain Shift Vulnerability:** MedGemma is trained on high-resolution, adult chest X-rays (e.g., MIMIC-CXR). Evaluating it on heavily downsampled pediatric MedMNIST images forces the model completely out of its training distribution, leading to heavy biases toward predicting "Normal."
* **The Super-Resolution Risk:** While implementing Swin2SR improved accuracy from 30% to 60%, introducing generative upscaling into a medical pipeline is a massive clinical risk. Super-resolution models rely on generative priors and can "hallucinate" textures to make an image look sharp. In a clinical setting, an AI upscaler might artificially generate textures that look exactly like pneumonia infiltrates, tricking the VLM into diagnosing an AI hallucination rather than the patient's actual anatomy.

![Swin2SR Hallucination Risk](https://raw.githubusercontent.com/AnasHXH/Task_2/main/SwinIr.png)
