import json
import pandas as pd
from pathlib import Path


DATA_PATH = Path("data/products.json")


def load_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found at: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("products.json must be a non-empty JSON array.")

    df = pd.DataFrame(data)
    return df