import faiss
import numpy as np
import os

def main():
    if not os.path.exists('data/features.npy'):
        print("Error: features.npy not found. Run extract_embeddings.py first.")
        return

    features = np.load('data/features.npy')
    embedding_dim = features.shape[1]

    print(f"Building FAISS index for {features.shape[0]} vectors...")
    
    # Using Inner Product (IP) because vectors are L2-normalized
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(features)

    faiss.write_index(index, 'data/medical_index.faiss')
    print("✅ FAISS index saved to data/medical_index.faiss")

if __name__ == "__main__":
    main()