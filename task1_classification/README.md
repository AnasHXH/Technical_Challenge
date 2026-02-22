# Task 1: CNN Classification with Comprehensive Analysis

## 1. Project Overview & Idea Description
This project tackles the binary classification of pneumonia from chest X-ray images using the PneumoniaMNIST dataset, a subset of the MedMNIST v2 benchmark. 

**The Challenge:** The dataset suffers from a significant class imbalance between the "Normal" and "Pneumonia" classes, which often causes standard deep learning models to overfit to the majority class (Pneumonia) and exhibit poor specificity.

**The Solution (Class Decomposition):** Instead of using standard oversampling or undersampling, I implemented a **Class Decomposition** strategy using K-Means clustering. 
1. **Feature Extraction:** A pre-trained visual language model (`openai/clip-vit-base-patch32`) was used to extract high-level semantic embeddings from all images in the majority "Pneumonia" class.
2. **Clustering:** K-Means clustering was applied to these embeddings to split the "Pneumonia" class into two distinct sub-types (`Pneumonia_Type_1` and `Pneumonia_Type_2`). [Image of K-Means clustering algorithm]
3. **Training Strategy:** The problem was transformed from a binary classification task to a 3-class classification task during training. This forced the model to learn finer-grained features within the pneumonia class rather than just generalizing it as a single block. During inference, the probabilities of the two pneumonia sub-types are summed to map the prediction back to the original binary label.

### Data Distribution Before and After Decomposition
By decomposing the majority class, the dataset distribution becomes much more balanced, preventing the model from collapsing into majority-class predictions.

**Before Decomposition (Binary):**
![Distribution Before](https://github.com/AnasHXH/Task_1/blob/main/results_decom/distribution_before.png)

**After Decomposition (3-Class):**
![Distribution After](https://github.com/AnasHXH/Task_1/blob/main/results_decom/distribution_after.png)

---

## 2. Repository Structure

```text
Task_1/
├── requirements.txt                   # Python dependencies required to run the scripts
├── README.md                          # Main repository overview
├── task1_classification_report.md     # This detailed methodology and results report
├── notebooks/
│   ├── Download_and_Analysis_dataset.ipynb # Data exploration and visualization
│   └── Tutorial.ipynb                 # End-to-end demonstration notebook
├── task1_classification/
│   ├── train_decomposition.py         # Script for CLIP feature extraction, clustering, and training
│   └── evaluate_decomp.py             # Script for testing the model and generating metrics
├── models/
│   └── best_model.pth                 # Saved weights for the trained MaxViT-T model
└── reports/                           # Generated evaluation outputs and figures
    ├── accuracy_curve.png
    ├── loss_curve.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── distribution_before.png
    ├── distribution_after.png
    ├── failure_cases.png
    └── metrics_summary.txt
```
---
## 3. Installation & Setup

**To reproduce the results, please install the required dependencies using the requirements.txt file. The primary requirements include PyTorch, torchvision, scikit-learn, transformers (for CLIP), and the official medmnist package.**
```
pip install -r requirements.txt
```
---
## 4. Download and Analysis Dataset


** Download the weight: [Google Drive Link](https://drive.google.com/file/d/1DFWhF_euBfmUaS-2XWgMnkZgsfxvvJc6/view?usp=sharing)

** Run Download_and_Analysis_dataset.ipynb

---
## 5. Train the Model:
```
python task1_classification/train_decomposition.py \
    --data_dir path/to/pneumoniamnist \
    --output_dir reports/ \
    --batch_size 4 \
    --epochs 32 \
    --learning_rate 0.0001
```
---
## 6. Evaluation the Model:
```
python task1_classification/evaluate_decomp.py \
    --data_dir path/to/pneumoniamnist \
    --model_path path/to/downloaded/best_model.pth \
    --output_dir reports/
```

---
## 7. Tutorial

** Run Tutorial.ipynb

---
## 8. Model Architecture & Training Methodology

**Architecture: MaxViT-T (Multi-Axis Vision Transformer). This hybrid architecture combines Convolutional Neural Networks (CNNs) for local feature extraction (e.g., small lung infiltrates) and Vision Transformers (ViTs) for global structural relationships (e.g., lung opacity).

Data Augmentation: Resize to 224x224, Random Rotation (10 degrees), Random Horizontal Flip, Color Jitter, and ImageNet standard normalization.

Loss Function: Weighted Cross-Entropy Loss. A weight of 2.0 was assigned to the 'Normal' class and 0.5 to the decomposed Pneumonia classes. 

Hyperparameters: Adam optimizer, learning rate of 0.0001, weight decay of 1e-4, batch size of 4, trained for 32 epochs. ReduceLROnPlateau was used for learning rate scheduling. **

---
## 9. Results & Evaluation Metrics

** The model achieved a best validation accuracy of 97.90% during training. Upon evaluation on the unseen test set using the binary mapping strategy, it achieved exceptional sensitivity. 

Final Test Metrics:

Accuracy: 0.8862

Precision: 0.8988

Recall: 0.8862

F1-Score: 0.8816

AUC: 0.9761

**confusion_matrix Test:**
![confusion_matrix Test](https://github.com/AnasHXH/Task_1/blob/main/results_test_decom/confusion_matrix.png)

**ROC Curve Test:**
![roc_curve](https://github.com/AnasHXH/Task_1/blob/main/results_test_decom/roc_curve.png)

**ROC Curve Test:**
![roc_curve](https://github.com/AnasHXH/Task_1/blob/main/results_test_decom/roc_curve.png)

---
## 10. Failure Case Analysis

** The confusion matrix indicates an over-prediction bias toward the Pneumonia class (67 False Positives vs. only 4 False Negatives). In a medical screening context, minimizing false negatives (missing a disease) is heavily preferred, though it comes at the cost of lower specificity. **

** Sample Misclassified Images: **
![failure_cases](https://github.com/AnasHXH/Task_1/blob/main/results_test_decom/failure_cases.png)

** Analysis of Errors: **

** A visual inspection of the failure cases (Normal images predicted as Pneumonia) reveals that the model struggles with normal vascular markings, pronounced rib shadows, or minor image blurring. Because the original dataset resolution is highly lightweight (28x28 pixels), upscaling to 224x224 for the MaxViT architecture introduces interpolation artifacts. The high sensitivity enforced by the weighted loss function causes the model to flag these ambiguous, interpolated structural densities as pneumonia infiltrates. While the system acts as an excellent and safe initial screening tool, it requires secondary clinical verification for positive flags.
