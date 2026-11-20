"""Real paths for the dev scripts, read from config/local/paths.yaml (git-ignored; example in config/local.example/)."""
from pathlib import Path
import yaml

_F = Path(__file__).resolve().parents[1] / "config" / "local" / "paths.yaml"
if not _F.exists():
    raise SystemExit(f"Missing {_F}: copy config/local.example/paths.yaml there and set your paths.")
P = yaml.safe_load(_F.read_text())["dev"]
