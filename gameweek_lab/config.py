from pathlib import Path

# Path del proyecto, calculado desde la ubicación de este archivo.
# Así los scripts funcionan sin importar desde qué carpeta los ejecutes.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FPL_BASE_URL = "https://fantasy.premierleague.com/api"
BOOTSTRAP_URL = f"{FPL_BASE_URL}/bootstrap-static/"
FIXTURES_URL = f"{FPL_BASE_URL}/fixtures/"

# CDN público de fotos de jugadores. Se arma con el campo 'code' de cada
# jugador (no viene como URL directa en la API). Tamaños disponibles:
# 40x40 (~15-25KB), 110x140 (~95-145KB), 250x250 (~300-350KB).
#
# Usamos 40x40: Tableau Public rechaza imágenes de más de 128KB, y
# 110x140 lo supera en algunos casos (depende de cuánto detalle tenga
# la foto, no solo del tamaño en píxeles) — 40x40 tiene margen de sobra
# siempre. Si no usás Tableau Public, se puede subir a 110x140 sin
# problema.
#
# Ojo: ~35% de los jugadores no tienen foto todavía (403 en cualquier
# tamaño) — típico en pre-temporada con fichajes nuevos sin sesión de
# fotos oficial. Se resuelve con el fallback genérico en photos.py, no
# hay forma de evitarlo desde acá — es un hueco real en los datos de
# origen.
PLAYER_PHOTO_URL_TEMPLATE = "https://resources.premierleague.com/premierleague/photos/players/40x40/p{code}.png"

# Silueta genérica que usa el propio sitio de FPL para jugadores sin foto
# todavía — la misma que verías en fantasy.premierleague.com.
FALLBACK_PHOTO_URL = "https://resources.premierleague.com/premierleague/photos/players/40x40/Photo-Missing.png"

SEASON = "2026-27"
