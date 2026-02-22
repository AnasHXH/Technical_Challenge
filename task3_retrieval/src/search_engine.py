import torch
import open_clip
import faiss
import numpy as np
from transformers import AutoTokenizer
from medmnist import PneumoniaMNIST

class MedicalSearchEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_id = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        
        # Load Model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_id)
        self.model = self.model.to(self.device).eval()
        
        # Load native tokenizer to avoid open_clip batch_encode_plus bug
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Load FAISS Index and Labels
        self.index = faiss.read_index('data/medical_index.faiss')
        self.labels = np.load('data/labels.npy')

    def search_by_text(self, query_text, k=5):
        tokens = self.tokenizer([query_text], padding=True, truncation=True, return_tensors="pt")
        text_tokens = tokens.input_ids.to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        
        distances, indices = self.index.search(text_features.cpu().numpy().astype('float32'), k)
        return indices[0], distances[0]

    def search_by_image(self, pil_image, k=5):
        img_tensor = self.preprocess(pil_image.convert('RGB')).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            img_features = self.model.encode_image(img_tensor)
            img_features /= img_features.norm(dim=-1, keepdim=True)
            
        distances, indices = self.index.search(img_features.cpu().numpy().astype('float32'), k)
        return indices[0], distances[0]

if __name__ == "__main__":
    # Example usage
    engine = MedicalSearchEngine()
    test_dataset = PneumoniaMNIST(split='test', size=28)
    
    print("\n--- Testing Image Search (Index 22) ---")
    img, _ = test_dataset[22]
    idxs, dists = engine.search_by_image(img)
    print(f"Top matches: {idxs}")

    print("\n--- Testing Text Search ---")
    idxs, dists = engine.search_by_text("Pneumonia infiltrates")
    print(f"Top matches: {idxs}")