"""
utils.py — Shared utility functions: logging setup, directory creation,
file I/O helpers, and timing decorators.

Windows-safe: The console handler uses UTF-8 encoding explicitly so that
special characters (Rs sign, arrows, box-drawing) do not crash on cp1252.
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from functools import wraps
from datetime import datetime

from src.config import LOGS_DIR, OUTPUTS_DIR, PLOTS_DIR, REPORTS_DIR, PREDICTIONS_DIR, METRICS_DIR, MODELS_DIR


# ── Logger Setup ──────────────────────────────────────────────────────────────

def get_logger(name: str = "retail_churn") -> logging.Logger:
    """
    Returns a logger that writes to both the terminal and a log file.
    The stream handler is forced to UTF-8 so Unicode chars work on Windows.
    """
    ensure_output_dirs()

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console handler — force UTF-8 so Rs/arrow/block chars don't crash cp1252
    try:
        stream = open(sys.stdout.fileno(), mode="w", encoding="utf-8",
                      buffering=1, closefd=False)
    except Exception:
        stream = sys.stdout

    ch = logging.StreamHandler(stream)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    log_file = LOGS_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ── Directory Management ──────────────────────────────────────────────────────

def ensure_output_dirs() -> None:
    for d in [OUTPUTS_DIR, PLOTS_DIR, REPORTS_DIR, PREDICTIONS_DIR,
              METRICS_DIR, LOGS_DIR, MODELS_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)


# ── Timing Decorator ──────────────────────────────────────────────────────────

def timed(logger=None):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.time()
            result = fn(*args, **kwargs)
            elapsed = time.time() - t0
            msg = f"{fn.__name__} completed in {elapsed:.2f}s"
            if logger:
                logger.info(msg)
            else:
                print(msg)
            return result
        return wrapper
    return decorator


# ── JSON Helpers ──────────────────────────────────────────────────────────────

def save_json(data: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=str)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Section Banner ────────────────────────────────────────────────────────────

def print_section(title: str, width: int = 60) -> None:
    bar = "=" * width
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")
