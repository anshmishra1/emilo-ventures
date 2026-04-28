import re
import pandas as pd


# =========================
# CONFIGURATION
# =========================

REQUIRED_COLUMNS = ["name", "description", "shortDescription", "slug"]

# Keyword → semantic enrichment terms added to combined_text
ENRICH_RULES = {
    "chocolate": "gift valentine birthday sweet",
    "flower":    "romantic love wedding anniversary",
    "hamper":    "festival diwali christmas gift",
    "watch":     "luxury gift accessory",
    "cake":      "birthday celebration party sweet",
    "jewelry":   "luxury anniversary wedding gift",
    "perfume":   "luxury gift romantic anniversary",
}

# Column name → how many times to repeat it for weighting
TEXT_WEIGHTS = {
    "name":             3,
    "shortDescription": 2,
    "description":      1,
    "slug":             1,
}


# =========================
# TEXT UTILITIES
# =========================

def safe_text(x) -> str:
    return "" if x is None else str(x)


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def enrich_text(text: str) -> str:
    for keyword, expansion in ENRICH_RULES.items():
        if keyword in text and expansion not in text:
            text += " " + expansion
    return text


# =========================
# MAIN PREPROCESSING
# =========================

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # Filter valid products
    df = df[(df["isApproved"] == True) & (df["status"] == "ACTIVE")].copy()

    # Handle missing values
    for col in REQUIRED_COLUMNS:
        df[col] = df[col].apply(safe_text)

    # Convert slug hyphens to spaces
    df["slug"] = df["slug"].str.replace("-", " ", regex=False)

    # Remove duplicates
    df = df.drop_duplicates(subset=["name", "description"])

    # Build weighted combined text
    df["combined_text"] = (
        df["name"] * TEXT_WEIGHTS["name"] + " " +
        df["shortDescription"] * TEXT_WEIGHTS["shortDescription"] + " " +
        df["description"] * TEXT_WEIGHTS["description"] + " " +
        df["slug"] * TEXT_WEIGHTS["slug"]
    )

    # Clean and enrich
    df["combined_text"] = df["combined_text"].apply(clean_text).apply(enrich_text)

    # Drop weak entries
    df = df[df["combined_text"].str.len() > 20].reset_index(drop=True)

    return df