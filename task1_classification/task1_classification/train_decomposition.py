

import os
import time
import copy
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from tqdm import tqdm

# NEW: Import HuggingFace Transformers for CLIP
from transformers import CLIPProcessor, CLIPVisionModel

def apply_class_decomposition(dataset, save_dir, device):
    """
    Splits the majority class (Pneumonia) into sub-classes using K-Means 
    clustering on extracted CLIP features to balance the dataset.
    """
    labels = [s[1] for s in dataset.samples]
    classes = dataset.classes
    counts_before = {c: labels.count(i) for i, c in enumerate(classes)}
    
    # Plot Distribution Before
    plt.figure(figsize=(8, 5))
    plt.bar(counts_before.keys(), counts_before.values(), color=['skyblue', 'salmon'])
    plt.title("Class Distribution (Before Decomposition)")
    plt.ylabel("Number of Images")
    plt.savefig(os.path.join(save_dir, 'distribution_before.png'))
    plt.close()

    print("\nApplying Class Decomposition to balance the dataset...")
    k_clusters = 2 # Changed to 2 as requested
    
    print(f"Extracting features to split Pneumonia into {k_clusters} clusters using CLIP...")
    
    # NEW: Load CLIP Processor and Vision Model
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    feature_extractor = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    feature_extractor.eval()
    
    majority_class_idx = 1 # Pneumonia
    majority_indices = [i for i, label in enumerate(labels) if label == majority_class_idx]
    features = []
    
    with torch.no_grad():
        for idx in tqdm(majority_indices, desc="Extracting Features"):
            img_path, _ = dataset.samples[idx]
            img = datasets.folder.default_loader(img_path)
            
            # Process image specifically for CLIP
            inputs = processor(images=img, return_tensors="pt").to(device)
            feat = feature_extractor(**inputs).pooler_output.cpu().numpy().squeeze()
            features.append(feat)
            
    print("Running K-Means clustering...")
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(features)
    
    # Modify Dataset Labels
    new_classes = ['Normal'] + [f'Pneumonia_Type_{i+1}' for i in range(k_clusters)]
    dataset.classes = new_classes
    dataset.class_to_idx = {c: i for i, c in enumerate(new_classes)}
    
    new_samples, new_targets = [], []
    maj_ptr = 0
    
    for path, label in dataset.samples:
        if label == 0: # Normal
            new_samples.append((path, 0))
            new_targets.append(0)
        else: # Pneumonia
            new_label = 1 + cluster_labels[maj_ptr] # Maps to 1 or 2
            new_samples.append((path, new_label))
            new_targets.append(new_label)
            maj_ptr += 1
            
    dataset.samples = new_samples
    dataset.targets = new_targets
    
    # Plot Distribution After
    counts_after = {c: new_targets.count(i) for i, c in enumerate(new_classes)}
    plt.figure(figsize=(10, 5))
    plt.bar(counts_after.keys(), counts_after.values(), color=sns.color_palette("pastel")[0:len(new_classes)])
    plt.title("Class Distribution (After Decomposition)")
    plt.ylabel("Number of Images")
    plt.savefig(os.path.join(save_dir, 'distribution_after.png'))
    plt.close()
    
    print(f"Decomposition complete! New distribution: {counts_after}\n")
    return dataset

def get_data_loaders(data_dir, batch_size, save_dir, device):
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=10),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    train_data = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(root=val_dir, transform=val_transforms)

    # Apply decomposition
    train_data = apply_class_decomposition(train_data, save_dir, device)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, valid_loader, len(train_data.classes)

def plot_curves(history, save_dir):
    history = np.array(history)
    
    plt.figure(figsize=(10, 6))
    plt.plot(history[:, 0], label='Training Loss')
    plt.plot(history[:, 1], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(history[:, 2], label='Training Accuracy')
    plt.plot(history[:, 3], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'accuracy_curve.png'))
    plt.close()

def train_model(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Data
    train_loader, valid_loader, num_classes = get_data_loaders(args.data_dir, args.batch_size, args.output_dir, device)

    # Model (Now outputs 3 classes: 1 Normal, 2 Pneumonia types)
    model = models.maxvit_t(weights=models.MaxVit_T_Weights.DEFAULT)
    model.classifier[5] = nn.Linear(512, num_classes)
    model = model.to(device)

    # Optimization
    # FIX: Ensure weights list matches exactly the number of classes (3)
    # Give higher weight to 'Normal' (Index 0), and standard weights to Pneumonia splits
    weight_list = [2.0] + [0.5] * (num_classes - 1) 
    weights = torch.tensor(weight_list).to(device) 
    
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    nll_loss = nn.NLLLoss()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = []

    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        train_loss, train_correct = 0.0, 0

        # --- TRAINING LOOP ---
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device, dtype=torch.long)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)

        # --- VALIDATION LOOP (Mapped back to Binary) ---
        model.eval()
        valid_loss, valid_correct = 0.0, 0

        with torch.no_grad():
            for inputs, labels in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                inputs, labels = inputs.to(device), labels.to(device, dtype=torch.long)
                outputs = model(inputs)
                
                # Convert multi-class outputs back to binary probabilities
                probs = torch.softmax(outputs, dim=1)
                prob_normal = probs[:, 0].unsqueeze(1)
                prob_pneumonia = probs[:, 1:].sum(dim=1).unsqueeze(1) 
                binary_probs = torch.cat([prob_normal, prob_pneumonia], dim=1)
                
                # Calculate Validation Loss using NLL on the binary mapping
                log_probs = torch.log(binary_probs + 1e-8)
                loss = nll_loss(log_probs, labels)
                
                valid_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(binary_probs, 1)
                valid_correct += torch.sum(preds == labels.data)

        # Calculate epoch metrics
        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = train_correct.double() / len(train_loader.dataset)
        epoch_valid_loss = valid_loss / len(valid_loader.dataset)
        epoch_valid_acc = valid_correct.double() / len(valid_loader.dataset)

        history.append([epoch_train_loss, epoch_valid_loss, epoch_train_acc.item(), epoch_valid_acc.item()])
        
        print(f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_valid_loss:.4f} Acc: {epoch_valid_acc:.4f}")

        scheduler.step(epoch_valid_loss)

        if epoch_valid_acc > best_acc:
            best_acc = epoch_valid_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    # Save outputs
    plot_curves(history, args.output_dir)
    torch.save(best_model_wts, os.path.join(args.output_dir, 'best_model.pth'))
    print(f"Training complete. Best Validation Accuracy: {best_acc:.4f}")
    print(f"Model and curves saved to {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Pneumonia Classification Model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Directory to save model and plots')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    args = parser.parse_args()
    
    train_model(args)
