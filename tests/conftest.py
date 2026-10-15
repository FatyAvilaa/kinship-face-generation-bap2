import sys
from pathlib import Path

# Permite `from src...` al ejecutar pytest desde cualquier carpeta.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
