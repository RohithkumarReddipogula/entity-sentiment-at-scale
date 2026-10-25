"""Tests never use the network or the user's home cache.

huggingface_hub reads these variables once, at import, so they must be set here, before any
test module imports transformers.
"""

import os
from pathlib import Path

os.environ["HF_HOME"] = str(Path(__file__).resolve().parents[1] / ".cache" / "hf")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
