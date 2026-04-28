import numpy as np
import re
from sentence_transformers import SentenceTransformer


# =========================
# CONFIGURATION
# =========================

QUERY_EXPANSIONS = {
    "birthday":    "birthday gift celebration cake party",
    "valentine":   "romantic love gift flowers chocolate",
    "diwali":      "festival lights sweets hampers gift",
    "wedding":     "marriage celebration luxury gift flowers",
    "anniversary": "romantic couple celebration gift",
    "office":      "corporate professional gift premium",
    "corporate":   "corporate professional gift premium formal",
    "romantic":    "romantic love couple flowers chocolate gift",
}

METADATA_BOOSTS = {
    ("flower",    "romantic"):  0.05,
    ("chocolate", "valentine"): 0.05,
    ("hamper",    "diwali"):    0.05,
    ("hamper",    "festival"):  0.05,
    ("cake",      "birthday"):  0.05,
}

INTENT_REASONS = {
    "romantic":    "Recommended for romantic occasions",
    "birthday":    "Perfect for birthday gifting",
    "diwali":      "Ideal for festive gifting",
    "wedding":     "Suitable for weddings and celebrations",
    "anniversary": "Great for anniversary gifting",
    "corporate":   "Ideal for corporate gifting",
}

SCORE_THRESHOLD = 0.2
MODEL_NAME = "all-MiniLM-L6-v2"


# =========================
# QUERY PROCESSING
# =========================

def clean_query(query: str) -> str:
    query = query.lower()
    query = re.sub(r"[^a-zA-Z0-9 ]", " ", query)
    return query.strip()


def expand_query(query: str) -> str:
    for trigger, expansion in QUERY_EXPANSIONS.items():
        if trigger in query and expansion not in query:
            query += " " + expansion
    return query


def process_query(query: str) -> str:
    return expand_query(clean_query(query))


# =========================
# MODEL BUILDING
# =========================

def build_embeddings(df) -> tuple:
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        df["combined_text"].tolist(),
        show_progress_bar=True,
        normalize_embeddings=True   # unit vectors → dot product == cosine similarity
    )
    return model, embeddings


# =========================
# SCORING
# =========================

def compute_scores(query_emb: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    # embeddings are already normalized; simple dot product gives cosine similarity
    return embeddings @ query_emb


def apply_metadata_boost(scores: np.ndarray, df, query: str) -> np.ndarray:
    scores = scores.copy()
    for (product_kw, query_kw), boost in METADATA_BOOSTS.items():
        if query_kw in query:
            mask = df["combined_text"].str.contains(product_kw, case=False, na=False)
            scores[mask.values] += boost
    return scores


# =========================
# EXPLANATION
# =========================

def generate_reason(query: str) -> str:
    for intent, reason in INTENT_REASONS.items():
        if intent in query:
            return reason
    return "Relevant to your search"


# =========================
# RECOMMENDATION
# =========================

def recommend_embeddings(query: str, model, embeddings: np.ndarray, df, top_k: int = 5) -> list[dict]:
    if not query.strip():
        return []

    query = process_query(query)

    query_emb = model.encode([query], normalize_embeddings=True)[0]
    scores = compute_scores(query_emb, embeddings)
    scores = apply_metadata_boost(scores, df, query)

    ranked_indices = scores.argsort()[::-1]
    reason = generate_reason(query)
    results = []
    seen_categories = set()

    for idx in ranked_indices:
        if len(results) >= top_k:
            break

        score = float(scores[idx])
        if score < SCORE_THRESHOLD:
            break  # sorted descending — safe to stop early

        product = df.iloc[idx]
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