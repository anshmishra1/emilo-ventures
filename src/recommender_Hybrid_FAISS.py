import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# CONFIGURATION
# =========================

INTENT_RULES = {
    "romantic":  ["girlfriend", "wife", "love", "romantic", "valentine"],
    "corporate": ["boss", "office", "colleague", "professional", "corporate"],
    "festival":  ["festival", "diwali", "holi", "christmas", "eid"],
    "birthday":  ["birthday", "bday", "born"],
    "wedding":   ["wedding", "marriage", "bride", "groom"],
    "anniversary": ["anniversary", "years together"],
}

QUERY_EXPANSIONS = {
    "birthday":    "birthday cake party celebration gifts",
    "valentine":   "romantic love couple flowers chocolate gift",
    "diwali":      "festival lights sweets hampers gift india",
    "wedding":     "marriage bride groom couple ceremony luxury gift flowers",
    "anniversary": "romantic couple love celebration gift",
    "office":      "corporate professional gift premium formal",
    "corporate":   "corporate professional gift premium formal",
    "romantic":    "romantic love couple flowers chocolate gift",
    "festival":    "festival celebration lights sweets hampers gift",
}

METADATA_BOOSTS = {
    # (keyword in product text, keyword in query): boost amount
    ("flower",     "romantic"):  0.05,
    ("chocolate",  "valentine"): 0.05,
    ("hamper",     "diwali"):    0.05,
    ("hamper",     "festival"):  0.05,
    ("cake",       "birthday"):  0.05,
    ("ring",       "wedding"):   0.05,
    ("ring",       "anniversary"): 0.05,
}

NEGATIVE_FILTERS = {
    # intent: [keywords in product text that should be penalized]
    "wedding":   ["holi", "diwali", "christmas"],
    "romantic":  ["corporate", "office"],
    "corporate": ["romantic", "wedding"],
    "birthday":  ["corporate", "office"],
}

GENERIC_KEYWORDS = ["gift wrapping", "gift cards", "gift box"]

SCORE_WEIGHTS = {"tfidf": 0.6, "embedding": 0.4}

SCORE_THRESHOLDS = {"general": 0.2, "default": 0.25}


# =========================
# 1. QUERY PROCESSING
# =========================

def clean_query(query: str) -> str:
    query = query.lower()
    query = re.sub(r"[^a-zA-Z0-9 ]", " ", query)
    return query.strip()


def detect_intent(query: str) -> str:
    query = query.lower()
    for intent, keywords in INTENT_RULES.items():
        if any(kw in query for kw in keywords):
            return intent
    return "general"


def expand_query(query: str) -> str:
    for trigger, expansion in QUERY_EXPANSIONS.items():
        if trigger in query and expansion not in query:
            query += " " + expansion
    return query


def expand_ambiguous_query(query: str) -> str:
    if len(query.split()) <= 2:
        query += " gift present item"
    return query


def process_query(query: str) -> str:
    query = clean_query(query)
    query = expand_query(query)
    query = expand_ambiguous_query(query)
    return query


# =========================
# 2. BUILD MODELS
# =========================

def build_models(df):
    # TF-IDF
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_vectors = tfidf.fit_transform(df["combined_text"])

    # Sentence embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["combined_text"].tolist(), show_progress_bar=True)

    # Normalize for cosine similarity via inner product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    # FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return tfidf, tfidf_vectors, model, index, embeddings


# =========================
# 3. SCORING
# =========================

def compute_tfidf_scores(query: str, tfidf, tfidf_vectors) -> np.ndarray:
    return cosine_similarity(
        tfidf.transform([query]),
        tfidf_vectors
    ).flatten()


def compute_embedding_scores(query: str, model, index, n: int) -> np.ndarray:
    query_emb = model.encode([query])
    norm = np.linalg.norm(query_emb, axis=1, keepdims=True)
    norm[norm == 0] = 1
    query_emb = query_emb / norm

    faiss_scores, faiss_indices = index.search(query_emb, n)

    emb_scores = np.zeros(n, dtype=float)
    for i, idx in enumerate(faiss_indices[0]):
        emb_scores[int(idx)] = float(faiss_scores[0][i])

    return emb_scores


def combine_scores(tfidf_scores: np.ndarray, emb_scores: np.ndarray) -> np.ndarray:
    return SCORE_WEIGHTS["tfidf"] * tfidf_scores + SCORE_WEIGHTS["embedding"] * emb_scores


# =========================
# 4. RANKING ENHANCEMENTS
# =========================

def apply_metadata_boost(scores: np.ndarray, df, query: str) -> np.ndarray:
    for (product_kw, query_kw), boost in METADATA_BOOSTS.items():
        if query_kw in query:
            mask = df["combined_text"].str.contains(product_kw, case=False, na=False)
            scores[mask.values] += boost
    return scores


def apply_negative_filter(scores: np.ndarray, df, intent: str) -> np.ndarray:
    penalized = NEGATIVE_FILTERS.get(intent, [])
    if penalized:
        pattern = "|".join(penalized)
        mask = df["combined_text"].str.contains(pattern, case=False, na=False)
        scores[mask.values] *= 0.3
    return scores


def penalize_generic_items(scores: np.ndarray, df) -> np.ndarray:
    pattern = "|".join(GENERIC_KEYWORDS)
    mask = df["combined_text"].str.contains(pattern, case=False, na=False)
    scores[mask.values] *= 0.6
    return scores


def generate_reason(intent: str) -> str:
    reasons = {
        "romantic":    "Romantic occasion",
        "corporate":   "Corporate gifting",
        "festival":    "Festive gifting",
        "birthday":    "Birthday gifting",
        "wedding":     "Wedding celebration",
        "anniversary": "Anniversary gifting",
        "general":     "General recommendation",
    }
    return reasons.get(intent, "Recommended for you")


# =========================
# 5. FINAL RECOMMENDER
# =========================

def recommend_hybrid_faiss(
    query: str,
    tfidf,
    tfidf_vectors,
    model,
    index,
    df,
    top_k: int = 5
) -> list[dict]:

    if not query or not query.strip():
        return []

    intent = detect_intent(query)
    processed_query = process_query(query)

    # Compute scores
    tfidf_scores = compute_tfidf_scores(processed_query, tfidf, tfidf_vectors)
    emb_scores = compute_embedding_scores(processed_query, model, index, len(df))
    final_scores = combine_scores(tfidf_scores, emb_scores)

    # Apply enhancements
    final_scores = apply_metadata_boost(final_scores, df, processed_query)
    final_scores = apply_negative_filter(final_scores, df, intent)
    final_scores = penalize_generic_items(final_scores, df)

    # Rank and filter
    threshold = SCORE_THRESHOLDS.get(intent, SCORE_THRESHOLDS["default"])
    ranked_indices = np.argsort(final_scores)[::-1]
    reason = generate_reason(intent)

    results = []
    seen_categories = set()

    for idx in ranked_indices:
        if len(results) >= top_k:
            break

        score = float(final_scores[idx])
        if score < threshold:
            break  # sorted descending, no point continuing

        product = df.iloc[int(idx)]
        category = product.get("categoryId", "")

        if category in seen_categories:
            continue
        seen_categories.add(category)

        results.append({
            "name":        str(product["name"]),
            "description": str(product["shortDescription"]),
            "image":       str(product["mainImage"]),
            "score":       round(score, 3),
            "reason":      reason,
        })

    return results