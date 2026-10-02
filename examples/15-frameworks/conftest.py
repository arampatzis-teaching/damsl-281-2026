"""Let the sessions in this folder import the modules that live beside them."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
