import sys
from pathlib import Path

# Permite ejecutar este archivo directamente (python scripts/run_fetch.py)
# sin importar desde qué carpeta lo llames.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gameweek_lab.fetch import fetch_all

if __name__ == "__main__":
    fetch_all()
