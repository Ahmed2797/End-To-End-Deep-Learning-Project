import os
import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

from src.chatbot.medicalai.helper import *


# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536, # OpenAI text-embedding-3-small
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

os.makedirs(IMAGE_DIR, exist_ok=True)
print("Image Directory created")
pdf_path = "src/chatbot/Data/braintumors_ucni.pdf"


def get_embedding(text):
    return client.embeddings.create(input=[text], model="text-embedding-3-small").data[0].embedding

def process_and_upsert(pdf_path):
    print('Open pdf.....')
    doc = fitz.open(pdf_path)
    
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            # 1. Extract Text
            text = page.extract_text()
            if text:
                vec = get_embedding(text)
                index.upsert([(f"p{i}_txt", vec, {"text": text, "type": "text", "page": i+1})])
            
            # 2. Extract Tables (Markdown format for LLM context)
            print("Extract Tables...")
            tables = page.extract_tables()
            for j, table in enumerate(tables):
                df = pd.DataFrame(table[1:], columns=table[0])
                table_md = df.to_markdown()
                vec = get_embedding(table_md)
                index.upsert([(f"p{i}_tbl_{j}", vec, {"text": table_md, "type": "table", "page": i+1})])

    # 3. Extract Images & Metadata
    print("Extract Images & Metadata...")

    for page_num in range(len(doc)):
        page = doc[page_num]
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            img_name = f"p{page_num+1}_img_{img_index}.png"
            pix.save(os.path.join(IMAGE_DIR, img_name))
            
            # Use page text as context for the image
            img_context = page.get_text("text")[:500]
            vec = get_embedding(img_context)
            print("# OpenAI text-embedding-3-small...")
            
            index.upsert([(f"p{page_num}_img_{img_index}", vec, {
                "text": f"Visual from page {page_num+1}: {img_context}",
                "type": "image",
                "file": img_name,
                "page": page_num+1
            })])

if __name__ =='__main__':
    print("# Run ingestion")
    process_and_upsert(pdf_path=pdf_path)