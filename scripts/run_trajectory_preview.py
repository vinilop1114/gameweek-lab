import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Consolas Windows con cp1252 crashean ante caracteres fuera de ese charset
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from gameweek_lab.analysis import add_expected_points, add_horizon_expected_points
from gameweek_lab.build_dataset import build_players_dataset
from gameweek_lab.config import DATA_PROCESSED_DIR
from gameweek_lab.transfer_advisor import simulate_squad_trajectory

if __name__ == "__main__":
    players = add_horizon_expected_points(add_expected_points(build_players_dataset()))
    trajectory, notes = simulate_squad_trajectory(players)

    print("\n=== Vista especulativa del equipo — próximas 4 fechas ===")
    print("(NO es una predicción: asume que nada cambia entre hoy y cada fecha simulada)\n")
    print(trajectory.to_string(index=False))
    print()
    for gw, note in notes.items():
        print(f"{gw}: {note}")

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED_DIR / "squad_trajectory_preview.csv"
    trajectory.to_csv(out_path, index=False)
    print(f"\nGuardado {out_path}")
