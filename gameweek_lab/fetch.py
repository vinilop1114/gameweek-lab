import json
from datetime import datetime, timezone

import requests

from gameweek_lab.config import BOOTSTRAP_URL, DATA_RAW_DIR, FIXTURES_URL


def _get_json(url: str) -> dict | list:
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    return response.json()


def _save_raw(payload: dict | list, name: str) -> None:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_RAW_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Guardado {path} ({len(payload)} elementos)")


def fetch_bootstrap() -> dict:
    data = _get_json(BOOTSTRAP_URL)
    _save_raw(data, "bootstrap-static")
    return data


def fetch_fixtures() -> list:
    data = _get_json(FIXTURES_URL)
    _save_raw(data, "fixtures")
    return data


def fetch_all() -> None:
    started = datetime.now(timezone.utc).isoformat()
    print(f"Descargando datos de FPL — {started}")
    fetch_bootstrap()
    fetch_fixtures()


if __name__ == "__main__":
    fetch_all()
