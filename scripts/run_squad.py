import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Consolas Windows con cp1252 crashean ante caracteres fuera de ese charset
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from gameweek_lab.squad_builder import recommend_squad

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Equipo Base autogestionado de The Gameweek Lab")
    parser.add_argument(
        "--force", action="store_true",
        help="Evalúa la transferencia aunque falte mucho para el deadline. "
             "Por defecto solo decide en las últimas horas previas, cuando ya "
             "se conocen las lesiones reportadas.",
    )
    args = parser.parse_args()

    recommend_squad(force=args.force)
