import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gameweek_lab.analysis import run_analysis

if __name__ == "__main__":
    run_analysis()
