import json
import pickle
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def load_chunks(json_path):    
    with open(json_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks")
    return chunks

def create_embeddings(chunks, model_name='all-MiniLM-L6-v2'):    
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print("Creating embeddings...")
    texts = [chunk['content'] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    
    print(f"Created embeddings with shape: {embeddings.shape}")
    return embeddings, model

def create_faiss_index(embeddings):    
    dimension = embeddings.shape[1]
        
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    
    print(f"FAISS index created with {index.ntotal} vectors")
    return index

def save_vector_store(index, chunks, model_name, output_dir='.'):    
    faiss.write_index(index, f"{output_dir}/constitution_faiss.index")
    print(f"Saved FAISS index to {output_dir}/constitution_faiss.index")
        
    with open(f"{output_dir}/constitution_metadata.json", 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved metadata to {output_dir}/constitution_metadata.json")
        
    with open(f"{output_dir}/model_info.txt", 'w') as f:
        f.write(model_name)
    print(f"Saved model info to {output_dir}/model_info.txt")

def test_search(index, model, chunks, query, top_k=3):    
    print(f"\nTesting search with query: '{query}'")
        
    query_vector = model.encode([query])
        
    distances, indices = index.search(query_vector.astype('float32'), top_k)
    
    print(f"\nTop {top_k} results:")
    for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
        print(f"\n{i+1}. Chunk {idx} (Distance: {distance:.4f})")
        print(f"Content: {chunks[idx]['content'][:200]}...")

if __name__ == "__main__":    
    CHUNKS_FILE = "../data/constitution_chunks.json"
    MODEL_NAME = 'all-MiniLM-L6-v2'
        
    chunks = load_chunks(CHUNKS_FILE)
        
    embeddings, model = create_embeddings(chunks, MODEL_NAME)
        
    index = create_faiss_index(embeddings)
        
    save_vector_store(index, chunks, MODEL_NAME)
        
    #test_search(index, model, chunks, "What are the fundamental rights?", top_k=3)
   