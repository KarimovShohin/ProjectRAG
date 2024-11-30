from sentence_transformers import SentenceTransformer
import faiss
import os
import numpy as np

def load_data(data_dir):
    texts = []
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            texts.append(f.read())
    return texts

def create_index(data_dir, index_file):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    texts = load_data(data_dir)
    
    if not texts:
        raise ValueError("No valid texts found in the data directory.")
    
    embeddings = model.encode(texts, show_progress_bar=True)
    
    if len(embeddings) == 0:
        raise ValueError("Embedding generation failed; check input texts or model.")
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, index_file)
    print(f"Index saved to {index_file}")

def load_index(index_file):
    if not os.path.exists(index_file):
        raise ValueError(f"Index file not found: {index_file}")
    
    index = faiss.read_index(index_file)
    print(f"Index loaded from {index_file}")
    return index

def search_index(index, model, query, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return distances, indices

if __name__ == "__main__":
    data_dir = "./data/processed"
    index_file = "./index/faiss_index"

    if not os.path.exists(index_file):
        create_index(data_dir, index_file)
    
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    index = load_index(index_file)
    
    query = "Who is Harry Poter?"
    top_k = 10
    distances, indices = search_index(index, model, query, top_k)
    
    texts = load_data(data_dir)
    
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        print(f"Result {i + 1}:")
        print(f"Text: {texts[idx]}")
        print(f"Distance: {dist}")
        print("-" * 50)
