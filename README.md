# End-To-End-Deep-Learning-Project & # AI-Medical-Assistant

This project implements a Deep Learning model for binary image classification.

***AI-powered medical chatbot using RAG architecture to provide accurate healthcare responses, built with LLMs, Pinecone vector database, and deployed on AWS.***

## step 1

### Clone the repository

    git clone https://github.com/Ahmed2797/End-To-End-Deep-Learning-Project.git

## step 2

### Create enverionment & install

    conda create -n tumar python=3.10 -y
    conda activate tumar

    pip install -r requirements.txt

## step 3

### Create a .env file in the root directory and add your Pinecone & openai credentials as follows

    PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxx"
    OPENAI_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxx"

    run:
    python store_pincone.py
    python main.py
    python app.py
