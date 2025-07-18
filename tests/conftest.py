import sys
import os

# Prepend project root so that `import src.*` works from tests/
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root not in sys.path:
    sys.path.insert(0, root)
