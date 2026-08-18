import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Consolas Windows con cp1252 crashean ante caracteres fuera de ese charset
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from gameweek_lab.analysis import add_expected_points, save_scored_players
from gameweek_lab.build_dataset import build_players_dataset
from gameweek_lab.photos import resolve_photo_urls
from gameweek_lab.squad_builder import export_squads_for_tableau

if __name__ == "__main__":
    players = add_expected_points(build_players_dataset())
    print("Verificando fotos de jugadores (puede tardar unos segundos)...")
    players = resolve_photo_urls(players)

    save_scored_players(players)
    export_squads_for_tableau(players)
