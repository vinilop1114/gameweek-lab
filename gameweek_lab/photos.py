from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from requests.adapters import HTTPAdapter

from gameweek_lab.config import FALLBACK_PHOTO_URL


def _make_session(pool_size: int) -> requests.Session:
    # Por default, requests abre una conexión TCP+TLS nueva por request si
    # no hay una sesión de por medio — con ~590 requests, eso se nota
    # mucho. Una Session con un pool de conexiones del tamaño de
    # max_workers reutiliza conexiones entre threads y es varias veces
    # más rápido.
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
    session.mount("https://", adapter)
    return session


def _photo_exists(session: requests.Session, url: str) -> bool:
    try:
        response = session.head(url, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def resolve_photo_urls(players: pd.DataFrame, max_workers: int = 50) -> pd.DataFrame:
    """Reemplaza por la silueta genérica de FPL las fotos que todavía no
    existen (~20% de los jugadores en pre-temporada — fichajes sin sesión
    oficial, ver README). No hay forma de saber de antemano quién tiene
    foto y quién no, así que se chequea cada URL una por una... pero en
    paralelo y reusando conexiones, porque son ~590 jugadores.
    """
    players = players.copy()
    urls = players["photo_url"].tolist()
    session = _make_session(max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        exists = list(pool.map(lambda url: _photo_exists(session, url), urls))

    players["photo_url"] = [url if ok else FALLBACK_PHOTO_URL for url, ok in zip(urls, exists)]
    return players
