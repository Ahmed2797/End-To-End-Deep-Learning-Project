import os
import fitz 
import pdfplumber
import pandas as pd
import shutil
import whisper
import sounddevice as sd
from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

app = FastAPI(title="RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. CONFIGURATION ---
IMAGE_DIR = "extracted_images"
UPLOAD_DIR = "uploads"
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/extracted_images", StaticFiles(directory=IMAGE_DIR), name="extracted_images")
templates = Jinja2Templates(directory="templates")

# --- 2. CLIENT SETUP ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

INDEX_NAME = "attention-paper-index-v1"
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME, 
        dimension=1536, 
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(INDEX_NAME)

def get_embedding(text):
    return client.embeddings.create(input=[text], model="text-embedding-3-small").data[0].embedding

@app.get("/", response_class=HTMLResponse)
async def front(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_and_process(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    doc = fitz.open(file_path)
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                index.upsert([(f"p{i}_txt", get_embedding(text), {"text": text, "type": "text", "page": i+1})])
            
            tables = page.extract_tables()
            for j, table in enumerate(tables):
                df = pd.DataFrame(table[1:], columns=table[0])
                table_md = df.to_markdown()
                index.upsert([(f"p{i}_tbl_{j}", get_embedding(table_md), {"text": table_md, "type": "table", "page": i+1})])

    for page_num in range(len(doc)):
        page = doc[page_num]
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            img_name = f"p{page_num+1}_img_{img_index}.png"
            pix.save(os.path.join(IMAGE_DIR, img_name))
            
            context = page.get_text("text")[:500]
            index.upsert([(f"p{page_num}_img_{img_index}", get_embedding(context), 
                           {"text": context, "type": "image", "file": img_name, "page": page_num+1})])

    return {"message": "Success! PDF indexed."}

@app.get("/ask")
async def ask_query(q: str = Query(...)):
    query_vec = get_embedding(q)
    results = index.query(vector=query_vec, top_k=3, include_metadata=True)
    
    if not results['matches']:
        raise HTTPException(status_code=404, detail="No relevant information found.")

    context_chunks = [match['metadata']['text'] for match in results['matches']]
    
    response = client.chat.completions.create(
        model="gpt-4.1-nano", 
        messages=[
            {"role": "system", "content": "You are a research assistant. Use the provided context to answer accurately."},
            {"role": "user", "content": f"Context: {' '.join(context_chunks)}\n\nQuestion: {q}"}
        ]
    )
    
    answer = response.choices[0].message.content
    image_url = None
    
    for match in results['matches']:
        if match['metadata'].get('type') == 'image':
            # FIX: Path now correctly matches the mount point
            image_url = f"/extracted_images/{match['metadata']['file']}"
            break

    return {"answer": answer, "relevant_image": image_url}


# --- 1. Move Model Loading to Global Scope (Top of file) ---
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")

# ... rest of your setup ...

@app.post("/voice-query")
async def handle_voice_query(file: UploadFile = File(...)):
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 2. Transcribe (Using the global whisper_model)
        result = whisper_model.transcribe(temp_file, fp16=False)
        user_text = result["text"].strip()

        if not user_text:
            return {"answer": "I couldn't hear anything. Please try again."}

        # 3. RAG Logic - Fixed the variable name from 'q' to 'user_text'
        query_vec = get_embedding(user_text)
        results = index.query(vector=query_vec, top_k=3, include_metadata=True)
        
        if not results['matches']:
            return {"answer": "No relevant info found.", "transcription": user_text}

        context_chunks = [match['metadata']['text'] for match in results['matches']]
        
        response = client.chat.completions.create(
            model="gpt-4.1-nano", # Note: Ensure your model name is correct (e.g., gpt-4o)
            messages=[
                {"role": "system", "content": "You are a research assistant. Use context to answer."},
                {"role": "user", "content": f"Context: {' '.join(context_chunks)}\n\nQuestion: {user_text}"} # Fixed 'q' to 'user_text'
            ]
        )
        
        answer = response.choices[0].message.content
        image_url = None
    
        for match in results['matches']:
            if match['metadata'].get('type') == 'image':
                image_url = f"/extracted_images/{match['metadata']['file']}"
                break

        return {
            "transcription": user_text, # Good for debugging
            "answer": answer, 
            "relevant_image": image_url
        }

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

## uvicorn app:app --reload