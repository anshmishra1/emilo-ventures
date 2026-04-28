import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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

SCORE_THRESHOLD = 0.1
TFIDF_MAX_FEATURES = 5000


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

def build_tfidf(df) -> tuple:
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=TFIDF_MAX_FEATURES
    )
    vectors = vectorizer.fit_transform(df["combined_text"])
    return vectorizer, vectors


# =========================
# RECOMMENDATION
# =========================

def recommend_tfidf(query: str, vectorizer, vectors, df, top_k: int = 5) -> list[dict]:
    if not query.strip():
        return []

    query = process_query(query)

    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, vectors).flatten()

    ranked_indices = scores.argsort()[::-1]
    results = []
    seen_names = set()

    for idx in ranked_indices:
        if len(results) >= top_k:
            break

        score = float(scores[idx])
        if score < SCORE_THRESHOLD:
            break  # sorted descending — safe to stop early

        product = df.iloc[idx]
        name = str(product["name"])

        if name in seen_names:
            continue
        seen_names.add(name)

        results.append({
            "name":        name,
            "description": str(product["shortDescription"]),
            "image":       str(product["mainImage"]),
            "score":       round(score, 3),
        })

    return results