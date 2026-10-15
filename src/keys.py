"""Load the api.data.gov key (used by USGS waterdata and EPA StreamCat) from the environment
or from the project's .env file, which may hold either `API_USGS_PAT=<key>` or the bare key."""
import os
from pathlib import Path

ENV = Path(__file__).resolve().parents[1] / ".env"


def api_key():
    if os.environ.get("API_USGS_PAT"):
        return os.environ["API_USGS_PAT"]
    if ENV.exists():
        text = ENV.read_text().strip()
        return text.split("=", 1)[1].strip().strip("'\"") if "=" in text else text
    return None
