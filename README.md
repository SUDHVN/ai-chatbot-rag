# AI Chatbot using RAG

## Description
This project is an AI chatbot built using Retrieval-Augmented Generation (RAG). 
It retrieves relevant information from documents and provides accurate responses.

## Features
- FastAPI backend
- Vector search using FAISS
- Sentence Transformers for embeddings
- Real-time query response

## How to Run
1. Install dependencies:
   pip install -r requirements.txt

2. Run server:
   uvicorn app:app --reload

3. Open browser:
   http://127.0.0.1:8000/docs

## Technologies Used
- Python
- FastAPI
- FAISS
- Sentence Transformers