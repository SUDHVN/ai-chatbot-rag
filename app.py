from fastapi import FastAPI, UploadFile, File, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pypdf import PdfReader
import re

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

documents = []
index = None

# ================= HOME (UI) =================
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ================= PDF UPLOAD =================
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global documents, index

    reader = PdfReader(file.file)
    text = ""

    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + " "

    # 🔥 TEXT CLEANING
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9., ]', '', text)

    # 🔥 CHUNKING (better than sentence split)
    words = text.split()
    chunk_size = 100

    documents = [
        " ".join(words[i:i+chunk_size])
        for i in range(0, len(words), chunk_size)
    ]

    # 🔥 EMBEDDINGS
    embeddings = model.encode(documents)

    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))

    return {"message": "PDF processed successfully"}


# ================= QUERY =================
class Query(BaseModel):
    question: str


@app.post("/ask")
def ask_question(query: Query):
    global index, documents

    if index is None:
        return {"answer": "Please upload a PDF first."}

    question = query.question.lower()

    # 🔥 SIMPLE SMART RESPONSES
    if question in ["hi", "hello", "hey"]:
        return {"answer": "Hello! Ask me something about the uploaded document."}

    # 🔥 VECTOR SEARCH
    q_embedding = model.encode([question])
    D, I = index.search(np.array(q_embedding), k=1)

    # 🔥 RELEVANCE CHECK
    if D[0][0] > 1.2:
        return {"answer": "I couldn't find relevant information in the document."}

    context = documents[I[0][0]]

    # 🔥 FORMATTED RESPONSE (ChatGPT-like)
    return {
        "answer": f"Based on the document:\n\n{context}"
    }