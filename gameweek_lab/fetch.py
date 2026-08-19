import json
from datetime import datetime, timezone

import requests

from gameweek_lab.config import BOOTSTRAP_URL, DATA_RAW_DIR, FIXTURES_URL, FPL_BASE_URL


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


def fetch_event_live(event_id: int) -> dict:
    """Puntos reales de cada jugador en UNA fecha específica — a
    diferencia de `event_points` en bootstrap-static (que solo refleja la
    fecha "actual" del juego y se pisa apenas arranca la siguiente), este
    endpoint no es ambiguo: siempre da los puntos de `event_id`, sin
    importar en qué fecha esté la temporada ahora. Se usa para completar
    el historial de calibración (gameweek_lab/calibration.py) — no se
    llama en cada corrida diaria, solo cuando hay una fecha finalizada
    pendiente de completar.
    """
    data = _get_json(f"{FPL_BASE_URL}/event/{event_id}/live/")
    _save_raw(data, f"event-{event_id}-live")
    return data


def fetch_all() -> None:
    started = datetime.now(timezone.utc).isoformat()
    print(f"Descargando datos de FPL — {started}")
    fetch_bootstrap()
    fetch_fixtures()


if __name__ == "__main__":
    fetch_all()
