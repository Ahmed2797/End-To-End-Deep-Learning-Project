import os
import io
import shutil
import tempfile
import logging
from typing import List
import numpy as np

import cv2
import fitz
import pdfplumber
import pandas as pd
import whisper

from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ultralytics import YOLO, SAM
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from dotenv import load_dotenv

from src.pipeline.prediction import ImagePredictor
# ------------------ CONFIG ------------------
load_dotenv()

UPLOAD_DIR = "uploads"
IMAGE_DIR = "extracted_images"
MODEL_PATH = "final_model/model.keras"
INDEX_NAME = "brain-tumors-ucni-v1"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)

# ------------------ APP INIT ------------------
app = FastAPI(
    title="AI Detection & Multi-Modal System",
    version="2.0"
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")
app.mount("/extracted_images", StaticFiles(directory=IMAGE_DIR), name="extracted_images")

templates = Jinja2Templates(directory="templates")

# ------------------ GLOBAL OBJECTS ------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index = None
yolo_model = None
sam_model = None
whisper_model = None
vgg_predictor = None


# ------------------ STARTUP ------------------
@app.on_event("startup")
def load_models():
    global index, vgg_predictor, yolo_model, sam_model, whisper_model

    logging.info("Loading models...")

    # Pinecone
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    index = pc.Index(INDEX_NAME)

    # Models
    vgg_predictor = ImagePredictor(MODEL_PATH)
    yolo_model = YOLO("models/computer-vision-brain_tumar-detection.pt")
    sam_model = SAM("models/sam_b.pt")
    whisper_model = whisper.load_model("base")

    logging.info("All models loaded successfully!")


# ------------------ UTIL ------------------
def get_embedding(text: str):
    return client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    ).data[0].embedding


# ------------------ FRONTEND ------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ------------------ PDF RAG ------------------
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if file.size and file.size > 10 * 1024 * 1024:
        raise HTTPException(400, "File too large")

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    doc = fitz.open(file_path)

    # TEXT + TABLES
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()

            if text:
                index.upsert([
                    (f"p{i}_txt", get_embedding(text),
                     {"text": text, "type": "text", "page": i + 1})
                ])

            tables = page.extract_tables()
            for j, table in enumerate(tables):
                df = pd.DataFrame(table[1:], columns=table[0])
                table_md = df.to_markdown()

                index.upsert([
                    (f"p{i}_tbl_{j}", get_embedding(table_md),
                     {"text": table_md, "type": "table", "page": i + 1})
                ])

    # IMAGES
    for page_num in range(len(doc)):
        page = doc[page_num]

        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)

            img_name = f"p{page_num+1}_img_{img_index}.png"
            img_path = os.path.join(IMAGE_DIR, img_name)

            pix.save(img_path)

            context = page.get_text("text")[:500]

            index.upsert([
                (f"p{page_num}_img_{img_index}",
                 get_embedding(context),
                 {
                     "text": context,
                     "type": "image",
                     "file": img_name,
                     "page": page_num + 1
                 })
            ])

    logging.info("PDF indexed successfully")

    return {"message": "PDF processed successfully"}


# ------------------ ASK (RAG) ------------------
@app.get("/ask")
async def ask(q: str = Query(...)):
    query_vec = get_embedding(q)

    results = index.query(vector=query_vec, top_k=3, include_metadata=True)

    if not results['matches']:
        raise HTTPException(404, "No relevant info found")

    context_chunks = [m['metadata']['text'] for m in results['matches']]
    context = "\n\n".join(context_chunks)

    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": "Answer using provided context."},
            {"role": "user", "content": f"{context}\n\nQuestion: {q}"}
        ]
    )

    answer = response.choices[0].message.content

    image_url = None
    for match in results['matches']:
        if match['metadata'].get("type") == "image":
            image_url = f"/extracted_images/{match['metadata']['file']}"
            break

    return {"answer": answer, "image": image_url}


# ------------------ VOICE ------------------
@app.post("/voice-query")
async def voice_query(file: UploadFile = File(...)):
    temp_file = f"temp_{file.filename}"

    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = whisper_model.transcribe(temp_file, fp16=False)
        text = result["text"].strip()

        if not text:
            return {"answer": "No speech detected"}

        return await ask(text)

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


# ------------------ VGG ---------------------
@app.post("/predict_vgg")
async def predict_vgg(file: UploadFile = File(...)):
    """Predict tumor using VGG16 model."""
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image file.")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            temp_file.write(await file.read())
            temp_path = temp_file.name

        label, confidence = vgg_predictor.predict(temp_path)

        return JSONResponse(content={
            "prediction": label,
            "confidence": confidence
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"VGG prediction failed: {str(e)}")

    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

# ------------------ YOLO DETECTION ------------------
@app.post("/detect")
async def detect(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image file.")

    # 1. Read the file bytes directly from the request
    file_bytes = await file.read()
    
    # 2. Convert bytes to a numpy array and decode to a cv2 image
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # 3. Pass the image array directly to YOLO
    results = yolo_model.predict(image)

    # 4. Draw your bounding boxes
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            if conf > 0.5:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 5. Encode and return the result
    _, img_encoded = cv2.imencode('.png', image)
    return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/png")


# ------------------ SAM SEGMENT ------------------
@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    # Read into memory
    file_bytes = await file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # YOLO can take the numpy array directly
    results = yolo_model.predict(image)

    boxes = []
    for r in results:
        for box in r.boxes:
            if float(box.conf[0]) > 0.5:
                boxes.append(box.xyxy[0].tolist())

    if not boxes:
        return JSONResponse(content={"message": "No tumor detected"}, status_code=200)

    # SAM also accepts the numpy array (image) directly
    sam_result = sam_model.predict(image, bboxes=boxes)[0]
    annotated = sam_result.plot()

    _, img_encoded = cv2.imencode('.png', annotated)
    return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/png")

# ------------------ HEALTH ------------------
@app.get("/api")
async def status():
    return {"message": "API running 🚀"}


# ------------------ RUN ------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

## uvicorn app:app --reload
