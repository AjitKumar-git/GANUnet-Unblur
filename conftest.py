import sys
from pathlib import Path

# Ensure root directory is in sys.path for test imports
root_dir = Path(__file__).parent.resolve()
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
