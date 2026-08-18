import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Consolas Windows con cp1252 crashean ante caracteres fuera de ese charset
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from gameweek_lab.analysis import run_analysis

if __name__ == "__main__":
    run_analysis()
