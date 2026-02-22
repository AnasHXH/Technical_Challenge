import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, precision_recall_fscore_support

def evaluate_model(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # Load Test Data
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_dir = os.path.join(args.data_dir, 'test')
    test_data = datasets.ImageFolder(root=test_dir, transform=test_transforms)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
    
    # The test folder only has ['Normal', 'Pneumonia']
    class_names = test_data.classes 

    # Load Model - Set to 4 classes because of the K-Means Decomposition
    trained_num_classes = 3
    
    model = models.maxvit_t(weights=None)
    model.classifier[5] = nn.Linear(512, trained_num_classes)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    all_preds, all_labels, all_probs = [], [], []

    # Inference Loop
    print("Running inference on test set with Binary Mapping...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            # Ensure labels are also long here just in case
            labels = labels.to(device, dtype=torch.long)
            outputs = model(inputs)
            
            # 1. Get raw probabilities across all 4 classes
            probs = torch.softmax(outputs, dim=1)
            
            # 2. Map 4 classes back to 2 (Binary Mapping)
            # Class 0 is Normal. Classes 1, 2, and 3 are sub-types of Pneumonia.
            prob_normal = probs[:, 0].unsqueeze(1)
            prob_pneumonia = probs[:, 1:].sum(dim=1).unsqueeze(1) # Sum the 3 Pneumonia probabilities
            binary_probs = torch.cat([prob_normal, prob_pneumonia], dim=1)
            
            # 3. Get the final predicted binary class
            _, preds = torch.max(binary_probs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(binary_probs[:, 1].cpu().numpy()) # Probability of Pneumonia for ROC

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Explicit Classification Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    metrics_text = (
        f"--- Final Evaluation Metrics (Decomposition Applied) ---\n"
        f"Accuracy:  {acc:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall:    {recall:.4f}\n"
        f"F1-Score:  {f1:.4f}\n"
        f"AUC:       {roc_auc:.4f}\n\n"
        f"--- Detailed Classification Report ---\n"
        f"{classification_report(all_labels, all_preds, target_names=class_names)}"
    )
    
    print(metrics_text)
    with open(os.path.join(args.output_dir, 'metrics_summary.txt'), 'w') as f:
        f.write(metrics_text)

    # 2. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=class_names, yticklabels=class_names, linewidths=2)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))
    plt.close()

    # 3. ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.04, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(args.output_dir, 'roc_curve.png'))
    plt.close()

    # 4. Identify Failure Cases
    print("\nExtracting failure cases...")
    fig, axes = plt.subplots(1, 5, figsize=(15, 4))
    fig.suptitle('Sample Failure Cases (Misclassified Images)', fontsize=14)
    
    misclassified_idx = [i for i, (p, t) in enumerate(zip(all_preds, all_labels)) if p != t]
    
    for i, idx in enumerate(misclassified_idx[:5]):
        if i >= 5: break
        img_path, _ = test_data.samples[idx]
        img = plt.imread(img_path)
        true_label = class_names[all_labels[idx]]
        pred_label = class_names[all_preds[idx]]
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}', color='red')
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'failure_cases.png'))
    plt.close()
    print(f"Evaluation complete. All visualizations and metrics saved to {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Pneumonia Classification Model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved best_model.pth')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Directory to save evaluation plots')
    args = parser.parse_args()
    
    evaluate_model(args)