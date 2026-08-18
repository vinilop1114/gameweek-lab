import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# La consola de Windows suele usar cp1252, que no tiene caracteres como
# '→' y crashea al imprimirlos. Forzamos UTF-8 en la salida del script en
# vez de depender de que cada usuario configure PYTHONIOENCODING.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from gameweek_lab.transfer_advisor import advise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Asesor de transferencias de The Gameweek Lab")
    parser.add_argument("--team", default="my_team.csv",
                        help="CSV con tu equipo: columna web_name (+ team_name opcional)")
    parser.add_argument("--bank", type=float, default=0.0, help="Dinero en el banco, en millones (ej. 1.5)")
    parser.add_argument("--free-transfers", type=int, default=1, help="Transferencias libres acumuladas (1-5)")
    args = parser.parse_args()

    advise(args.team, bank=args.bank, free_transfers=args.free_transfers)
