from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from src.data_loader import load_data
from src.processing import preprocess_data
from src.recommender_Hybrid_FAISS import build_models, recommend_hybrid_faiss

import os


# Initialize app
app = FastAPI()

# Templates
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
# print("Template dir:", os.path.join(BASE_DIR, "templates"))
# templates = Jinja2Templates(directory="app/templates")

# Load + preprocess data (once)
df = preprocess_data(load_data())

# Build models (once)
tfidf, tfidf_vectors, model, index, embeddings = build_models(df)


# ------------------------
# Home Route
# ------------------------
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "results": None,
            "query": ""
        }
    )

@app.post("/search")
def search(request: Request, occasion: str = Form(...)):
    results = recommend_hybrid_faiss(
        occasion, tfidf, tfidf_vectors, model, index, df
    )
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "results": results,
            "query": occasion
        }
    )