# Pinecone Index Setup
INDEX_NAME = "brain-tumors-ucni-v1"

# Image-Dir
IMAGE_DIR = "extracted_images"

# Pdf-Path
pdf_path = "src/chatbot/Data/braintumors_ucni.pdf"

prompt_template = """
You are a helpful Medical AI assistant. Answer questions based on the provided context.

CONTEXT:
{context}

QUESTION:
{input}

INSTRUCTIONS:
1. Use only information from the context
2. If context doesn't contain answer, say "I don't know"
3. Keep answer concise (4-6 sentences)
4. Use simple language

ANSWER:
"""

