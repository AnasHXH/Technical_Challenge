# Task 1: CNN Classification with Comprehensive Analysis

## 1. Project Overview & Idea Description
This project tackles the binary classification of pneumonia from chest X-ray images using the PneumoniaMNIST dataset.

**The Challenge:** The dataset suffers from a significant class imbalance between the "Normal" and "Pneumonia" classes, which often causes standard deep learning models to overfit to the majority class (Pneumonia) and exhibit poor specificity.

**The Solution (Class Decomposition):** Instead of using standard oversampling or undersampling, I implemented a **Class Decomposition** strategy using K-Means clustering. 
1. **Feature Extraction:** A pre-trained visual language model (`openai/clip-vit-base-patch32`) was used to extract high-level semantic embeddings from all images in the majority "Pneumonia" class.
2. **Clustering:** K-Means clustering was applied to these embeddings to split the "Pneumonia" class into two distinct sub-types (`Pneumonia_Type_1` and `Pneumonia_Type_2`). 
3. **Training Strategy:** The problem was transformed from a binary classification task to a 3-class classification task during training. This forced the model to learn finer-grained features within the pneumonia class rather than just generalizing it as a single block. During inference, the probabilities of the two pneumonia sub-types are summed to map the prediction back to the original binary label.

### Data Distribution Before and After Decomposition
By decomposing the majority class, the dataset distribution becomes much more balanced.

**Before Decomposition (Binary):** ![Distribution Before](https://github.com/AnasHXH/Task_1/blob/main/reports/distribution_before.png)

**After Decomposition (3-Class):** ![Distribution After](https://github.com/AnasHXH/Task_1/blob/main/reports/distribution_after.png)

---

## 2. Model Architecture & Training Methodology

* **Architecture:** **MaxViT-T** (Multi-Axis Vision Transformer). This hybrid architecture combines Convolutional Neural Networks (CNNs) for local feature extraction (e.g., small lung infiltrates) and Vision Transformers (ViTs) for global structural relationships (e.g., lung opacity).
* **Data Augmentation:** Resize to 224x224, Random Rotation (10 degrees), Random Horizontal Flip, Color Jitter, and ImageNet standard normalization.
* **Loss Function:** Weighted Cross-Entropy Loss. A weight of 2.0 was assigned to the 'Normal' class and 0.5 to the decomposed Pneumonia classes. 
* **Hyperparameters:** Adam optimizer, learning rate of 0.0001, weight decay of 1e-4, batch size of 4, trained for 32 epochs. `ReduceLROnPlateau` was used for learning rate scheduling.

---

## 3. Results & Evaluation Metrics

The model achieved a best validation accuracy of 97.90% during training. Upon evaluation on the unseen test set using the binary mapping strategy, it achieved exceptional sensitivity. 

### Final Test Metrics:
* **Accuracy:** 0.8862
* **Precision:** 0.8988
* **Recall:** 0.8862
* **F1-Score:** 0.8816
* **AUC:** 0.9761

### Training Curves
![Loss Curve](https://github.com/AnasHXH/Task_1/blob/main/reports/loss_curve.png)
![Accuracy Curve](https://github.com/AnasHXH/Task_1/blob/main/reports/accuracy_curve.png)

### Receiver Operating Characteristic (ROC) & Confusion Matrix
The model successfully identified 386 out of 390 positive pneumonia cases, achieving a 99% recall rate for the positive class.

![Confusion Matrix](https://github.com/AnasHXH/Task_1/blob/main/reports/confusion_matrix.png)
![ROC Curve](https://github.com/AnasHXH/Task_1/blob/main/reports/roc_curve.png)

---

## 4. Failure Case Analysis

The confusion matrix indicates an over-prediction bias toward the Pneumonia class (67 False Positives vs. only 4 False Negatives). In a medical screening context, minimizing false negatives (missing a disease) is heavily preferred, though it comes at the cost of lower specificity. 

**Sample Misclassified Images:** ![Failure Cases](https://github.com/AnasHXH/Task_1/blob/main/reports/failure_cases.png)

**Analysis of Errors:** A visual inspection of the failure cases (Normal images predicted as Pneumonia) reveals that the model struggles with normal vascular markings, pronounced rib shadows, or minor image blurring. Because the original dataset resolution is highly lightweight (28x28 pixels), upscaling to 224x224 for the MaxViT architecture introduces interpolation artifacts. The high sensitivity enforced by the weighted loss function causes the model to flag these ambiguous, interpolated structural densities as pneumonia infiltrates. While the system acts as an excellent and safe initial screening tool, it requires secondary clinical verification for positive flags.
