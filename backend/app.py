import json
import faiss
import os
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from groq import Groq
from models import Query, Source, Response


app = FastAPI(title="Aaeen Bot")

groq_client = None
embedding_model = None
index = None
chunks = None

@app.on_event("startup")
async def load_resources():    
    global index, chunks, embedding_model, groq_client    
    try:        
        index = faiss.read_index("../embeddings/constitution_faiss.index")
                
        with open("../embeddings/constitution_metadata.json", 'r', encoding='utf-8') as f:
            chunks = json.load(f)
                
        with open("../embeddings/model_info.txt", 'r') as f:
            model_name = f.read().strip()
                
        embedding_model = SentenceTransformer(model_name)
                
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        
        groq_client = Groq(api_key=groq_api_key)
        
        print(f"Loaded {len(chunks)} chunks and initialized Groq client")
        
    except Exception as e:
        print(f"Error loading resources: {e}")
        raise

@app.get("/")
async def root():    
    return {
        "status": "ok",
        "message": "Aaeen Bot",
        "total_chunks": len(chunks) if chunks else 0
    }

@app.post("/query", response_model=Response)
async def query_constitution(query: Query):
    if not query.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:        
        relevant_chunks = retrieve_chunks(query.question, query.top_k)
                
        context = "\n\n".join([f"Section {i+1}:\n{chunk['content']}" 
                              for i, chunk in enumerate(relevant_chunks)])
                
        answer = generate_answer(query.question, context, query.max_tokens)
                
        sources = [Source(**chunk) for chunk in relevant_chunks]
        
        return Response(
            question=query.question,
            answer=answer,
            sources=sources
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



def retrieve_chunks(question: str, top_k: int = 3):    
    query_vector = embedding_model.encode([question])
    
    distances, indices = index.search(query_vector.astype('float32'), top_k)
    
    relevant_chunks = []
    for idx, distance in zip(indices[0], distances[0]):
        relevant_chunks.append({
            'chunk_id': int(idx),
            'content': chunks[idx]['content'],
            'distance': float(distance)
        })
    
    return relevant_chunks



def generate_answer(question: str, context: str, max_tokens: int = 500):    
    prompt = f"""You are an expert on the Constitution of Pakistan. Answer the user's question based on the provided context from the constitution.

                Context from Constitution:
                {context}

                Question: {question}

                Instructions:
                - Answer based ONLY on the provided context
                - Be accurate and cite specific parts when possible
                - If the context doesn't contain enough information, say so
                - Keep answers clear and concise

                Answer:"""

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert assistant on Pakistan's Constitution."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=max_tokens,
        )
        
        return chat_completion.choices[0].message.content
        
    except Exception as e:
        print(e)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)