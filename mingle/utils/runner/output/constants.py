from pathlib import Path
import re

REPRESENTATION_DIR = "representation"
BENCHMARKS_DIR = "benchmarks"
ALLOCATION_DIR = "allocations"

CONFIG_FILE = "config.yaml"
METADATA_FILE = "metadata.json"
WINDOW_FILE = "window_info.json"
SUMMARY_FILE = "summary.csv"
HISTORY_FILE = "history.csv"

F_FILE = "F.npy"
B_FILE = "B.npy"
Y_FILE = "Y.npy"
W_FILE = "W.npy"

IN_RETURNS_FILE = "in_returns.csv"
OUT_RETURNS_FILE = "out_returns.csv"
WEIGHTS_FILE = "weights.csv"
PERFORMANCE_FILE = "performance.json"
RETURN_COL = "return"

WINDOWS_DIR = Path("windows")
WINDOW_PREFIX = "window"
WINDOW_DIGITS = 4
WINDOW_NAME_FMT = f"{WINDOW_PREFIX}_{{idx:0{WINDOW_DIGITS}d}}"

WINDOWS_RE = re.compile(
    rf"^{WINDOW_PREFIX}_(\d{{{WINDOW_DIGITS}}})$"
)

def get_window_name(window_idx: int) -> str:
    return WINDOW_NAME_FMT.format(idx=window_idx)

def get_window_dir(window_idx: int) -> Path:
    return WINDOWS_DIR / get_window_name(window_idx)
