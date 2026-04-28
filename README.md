# 🎁 Gift Concierge — Occasion-Based Product Recommendation Engine

A full-stack AI application that takes a natural language occasion query and returns semantically ranked product recommendations using a hybrid NLP pipeline.

Built with FastAPI, Sentence Transformers, FAISS, and TF-IDF — served through a clean Jinja2 frontend.

---

## 🧠 The Problem

Generic product search fails for gifting. A user searching *"something for Diwali"* or *"gift for my wife's anniversary"* doesn't want keyword matches — they want contextually relevant, ranked suggestions that understand **intent**, not just words.

This project solves that with a hybrid retrieval pipeline that blends lexical and semantic understanding.

---

## ⚙️ How It Works

### Stage 1 — Query Understanding

Raw queries are noisy and ambiguous. Before any scoring happens, the query goes through three steps:

- **Cleaning** — lowercased, punctuation stripped
- **Intent Detection** — classifies the query into intents (`romantic`, `corporate`, `festival`, `birthday`, `wedding`, `general`) using a config-driven keyword map
- **Expansion** — short or ambiguous queries are enriched with semantically related terms (e.g. `"diwali"` → `"festival lights sweets hampers gift india"`) so both retrievers have enough signal to work with

### Stage 2 — Hybrid Scoring

Two independent scores are computed and blended:

**TF-IDF (lexical, weight: 60%)**
Captures exact keyword overlap between the expanded query and the product's `combined_text`. Fast and precise for well-defined occasions.

**Sentence Embeddings + FAISS (semantic, weight: 40%)**
The query and all product texts are encoded using `all-MiniLM-L6-v2`. Embeddings are L2-normalised so FAISS inner-product search is equivalent to cosine similarity. This captures meaning even when exact keywords don't match.

The final score is a weighted blend:
```
final_score = 0.6 × tfidf_score + 0.4 × embedding_score
```

### Stage 3 — Ranking Enhancements

Raw scores are adjusted before final ranking:

- **Metadata Boosts** — products containing contextually strong signals for the detected intent get a score bump (e.g. `flower` + `romantic` query → +0.05)
- **Negative Filters** — products semantically mismatched to the intent are penalised (e.g. `corporate` products for a `romantic` query → ×0.3)
- **Generic Item Penalty** — items like gift cards or gift wrapping are deprioritised (×0.6) since they are weak recommendations regardless of occasion
- **Category Deduplication** — the top-K results are enforced to span distinct product categories, preventing the same type of product dominating all slots

All of the above is config-driven — boosts, filters, thresholds, and weights live as plain dicts at the top of the file, not buried in logic.

---

## 🗂️ Project Structure

```
Emilo/
├── app/
│   └── app.py                          # FastAPI routes, model loading, request handling
├── src/
│   ├── data_loader.py                  # Loads and validates products.json
│   ├── processing.py                   # Text cleaning, enrichment, combined_text construction
│   ├── recommender_Hybrid_FAISS.py     # Primary pipeline (TF-IDF + Embeddings + FAISS)
│   ├── recommender_Embedding.py        # Standalone semantic recommender
│   └── recommender_TFIDF.py            # Standalone lexical recommender
├── data/
│   └── products.json                   # Product catalogue
├── templates/
│   └── index.html                      # Jinja2 frontend template
├── requirements.txt
└── README.md
```

---

## 🔬 Data Pipeline

Products are loaded from a flat JSON catalogue and preprocessed into a single `combined_text` field before any model sees them:

- Invalid or inactive products are filtered out (`isApproved`, `status`)
- Text fields are weighted before concatenation — `name` ×3, `shortDescription` ×2, `description` ×1, `slug` ×1 — so the model naturally prioritises product names in scoring
- Semantic enrichment is applied: products containing `"flower"` get `"romantic love wedding anniversary"` appended, products with `"hamper"` get `"festival diwali christmas gift"`, and so on — this bridges the gap between product copy language and how users actually phrase occasion queries
- Duplicate products are removed on `name + description` before indexing

---

## 🧩 Recommender Comparison

Three recommenders were built and evaluated:

| | TF-IDF | Sentence Embedding | Hybrid FAISS |
|---|---|---|---|
| Keyword precision | ✅ Strong | ❌ Weak | ✅ |
| Semantic understanding | ❌ | ✅ Strong | ✅ |
| Handles ambiguous queries | ❌ | ✅ | ✅ |
| Category diversity | ❌ | ✅ | ✅ |
| Metadata & intent boosting | ❌ | ✅ | ✅ |
| Deployed in production route | ❌ | ❌ | ✅ |

The hybrid approach was chosen for the production route because neither retriever alone handles the full range of query types — TF-IDF fails on semantically phrased queries, embeddings miss exact keyword matches on niche occasions.

---

## 🛠️ Stack

| Layer | Technology |
|---|---|
| Web framework | FastAPI |
| Frontend templating | Jinja2 |
| Lexical retrieval | scikit-learn TF-IDF |
| Semantic retrieval | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector search | FAISS (`IndexFlatIP`) |
| Data processing | pandas, numpy |
| Server | Uvicorn |

---

*Built for an AI/ML Product Recommendation assignment © 2026*
