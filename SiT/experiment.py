"""Shared experiment contract used by the trainer and the read-only dashboard."""
import json
from pathlib import Path

STEPS = (1, 3, 5)
TEMPS = (0.0, 0.3, 0.6)
CLASSES = (207, 360, 387, 974)
BATCH = 128
LR = 0.0001
INTERVAL = 1800
UNIT = "va-diffusion-sit-s2-train.service"
RUN = Path(__file__).resolve().parents[1] / "runs" / "imagenet_sit_s2_100m"


def beta(completed):
    return max(1e-4, min(1.0, (completed - 21) / (21 + 5000)))


def grid_name(nfe, temperature, slot):
    return f"n{nfe}_t{round(10*temperature):02d}_{slot}.png"


def write_json(path, value):
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def read_json(path, fallback=None):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return fallback
