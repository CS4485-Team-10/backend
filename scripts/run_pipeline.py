import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable

if __name__ == "__main__":
    subprocess.run([PYTHON, str(SCRIPTS_DIR / "misinfo_checker.py"), "3"], cwd=SCRIPTS_DIR)
    subprocess.run([PYTHON, str(SCRIPTS_DIR / "misinfo_checker.py"), "4"], cwd=SCRIPTS_DIR)
    subprocess.run([PYTHON, str(SCRIPTS_DIR / "sentiment_analysis.py"), "2"], cwd=SCRIPTS_DIR)
