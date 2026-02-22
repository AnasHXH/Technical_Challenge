import torch
import open_clip
import numpy as np
import os
from medmnist import PneumoniaMNIST
from tqdm import tqdm

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    
    print(f"Loading BiomedCLIP on {device}...")
    model, _, preprocess_val = open_clip.create_model_and_transforms(model_id)
    model = model.to(device).eval()

    print("Loading dataset...")
    test_dataset = PneumoniaMNIST(split='test', download=True, size=28)
    
    features = []
    labels = []

    print("Extracting embeddings...")
    with torch.no_grad():
        for i in tqdm(range(len(test_dataset))):
            img, label = test_dataset[i]
            img_rgb = img.convert('RGB')
            img_tensor = preprocess_val(img_rgb).unsqueeze(0).to(device)
            
            # Extract and normalize
            img_features = model.encode_image(img_tensor)
            img_features /= img_features.norm(dim=-1, keepdim=True)
            
            features.append(img_features.cpu().numpy().squeeze())
            labels.append(label[0])

    # Save for the next stage
    os.makedirs('data', exist_ok=True)
    np.save('data/features.npy', np.array(features).astype('float32'))
    np.save('data/labels.npy', np.array(labels))
    print("✅ Embeddings saved to data/features.npy")

if __name__ == "__main__":
    main()