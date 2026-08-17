import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gameweek_lab.squad_builder import recommend_wildcard_squad

if __name__ == "__main__":
    recommend_wildcard_squad()
