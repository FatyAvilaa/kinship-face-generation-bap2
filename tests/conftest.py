import sys
from pathlib import Path

# Allows `from src...` when pytest is run from any folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
