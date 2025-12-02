import PyPDF2
import json
import re

def extract_text_from_pdf(pdf_path):    
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            print(f"Total pages: {len(pdf_reader.pages)}")
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page_text
                
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

def clean_text(text):    
    text = re.sub(r'\s+', ' ', text)    
    text = re.sub(r'Page \d+', '', text)
    return text.strip()

def chunk_text(text, chunk_size=1000, overlap=200):    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
                
        if end < text_length:            
            last_period = text.rfind('. ', start, end)
            if last_period != -1 and last_period > start + chunk_size // 2:
                end = last_period + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({
                'chunk_id': len(chunks),
                'start_char': start,
                'end_char': end,
                'content': chunk
            })
        
        start = end - overlap
    
    return chunks

def save_to_json(data, output_path):    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Data saved to {output_path}")

def save_to_txt(text, output_path):    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Text saved to {output_path}")

if __name__ == "__main__":    
    pdf_path = "constitution.pdf"
        
    raw_text = extract_text_from_pdf(pdf_path)
    
    if raw_text:        
        save_to_txt(raw_text, "raw_constitution_text.txt")                
        cleaned_text = clean_text(raw_text)
        save_to_txt(cleaned_text, "cleaned_constitution_text.txt")        
        chunks = chunk_text(cleaned_text, chunk_size=1000, overlap=200)        
        save_to_json(chunks, "constitution_chunks.json")
        print("Constitution processed successfully.")
    else:
        print("Failed to extract text from PDF")