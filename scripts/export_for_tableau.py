import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Consolas Windows con cp1252 crashean ante caracteres fuera de ese charset
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from gameweek_lab.analysis import (
    add_ceiling_metrics,
    add_expected_points,
    add_horizon_expected_points,
    save_last_season_baseline,
    save_scored_players,
)
from gameweek_lab.briefing import save_briefing
from gameweek_lab.build_dataset import build_players_dataset
from gameweek_lab.photos import resolve_photo_urls
from gameweek_lab.squad_builder import export_squads_for_tableau
from gameweek_lab.transfer_advisor import evolve_base_squad

if __name__ == "__main__":
    raw_players = build_players_dataset()

    # Antes de calcular nada: congelar la titularidad de la temporada
    # anterior mientras `starts` todavía la refleja. Es idempotente y no
    # hace nada una vez arrancada la temporada.
    print(save_last_season_baseline(raw_players))

    players = add_ceiling_metrics(add_horizon_expected_points(add_expected_points(raw_players)))

    # Fotos resueltas ANTES de evolucionar el Base: evolve_base_squad toma
    # una porción (slice) del DataFrame, así que si resolviéramos las fotos
    # después, esa porción quedaría con las URLs viejas sin verificar.
    print("Verificando fotos de jugadores (puede tardar unos segundos)...")
    players = resolve_photo_urls(players)
    save_scored_players(players)

    base_starters, base_bench, log = evolve_base_squad(players)
    print("\n=== Equipo Base — evolución automática ===")
    for line in log:
        print(f"  {line}")

    squads = export_squads_for_tableau(players, base_starters, base_bench)
    save_briefing(players, squads)
