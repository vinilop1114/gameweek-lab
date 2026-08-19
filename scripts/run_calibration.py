import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Consolas Windows con cp1252 crashean ante caracteres fuera de ese charset
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from gameweek_lab.analysis import add_expected_points
from gameweek_lab.build_dataset import build_players_dataset
from gameweek_lab.calibration import build_calibration_report, record_actual_points, snapshot_predictions

if __name__ == "__main__":
    players = add_expected_points(build_players_dataset())

    print(snapshot_predictions(players))
    print(record_actual_points())
    print()
    print(build_calibration_report())
