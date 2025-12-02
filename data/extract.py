import json
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using LangChain"""
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        print(f"Total pages: {len(pages)}")
        
        # Combine all pages with page markers
        text = ""
        for page in pages:
            page_num = page.metadata.get('page', 0) + 1
            text += f"\n--- Page {page_num} ---\n"
            text += page.page_content
        
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

def clean_text(text):
    """Clean extracted text"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'Page \d+', '', text)
    return text.strip()

def chunk_text_langchain(text, chunk_size=500, overlap=200):
    """Chunk text using LangChain RecursiveCharacterTextSplitter"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Split text into chunks
    chunks_text = text_splitter.split_text(text)
    
    # Convert to dictionary format
    chunks = []
    for i, chunk in enumerate(chunks_text):
        chunks.append({
            'chunk_id': i,
            'content': chunk,
            'length': len(chunk)
        })
    
    return chunks

def save_to_json(data, output_path):
    """Save data to JSON"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Data saved to {output_path}")

def save_to_txt(text, output_path):
    """Save text to file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Text saved to {output_path}")

if __name__ == "__main__":
    pdf_path = "constitution.pdf"
    
    print("Step 1: Extracting text from PDF using LangChain...")
    raw_text = extract_text_from_pdf(pdf_path)
    
    if raw_text:
        save_to_txt(raw_text, "raw_constitution_text.txt")
        
        print("\nStep 2: Cleaning text...")
        cleaned_text = clean_text(raw_text)
        save_to_txt(cleaned_text, "cleaned_constitution_text.txt")
        
        print("\nStep 3: Chunking with LangChain...")
        chunks = chunk_text_langchain(cleaned_text, chunk_size=500, overlap=200)
        
        print(f"Created {len(chunks)} chunks")
        
        # Show sample
        print("\nSample chunks:")
        for i in [0, len(chunks)//2, -1]:
            print(f"\nChunk {chunks[i]['chunk_id']} (length: {chunks[i]['length']}):")
            print(f"{chunks[i]['content'][:200]}...")
        
        save_to_json(chunks, "constitution_chunks.json")
        
        print("\n✅ Constitution processed successfully with LangChain!")
    else:
        print("Failed to extract text from PDF")