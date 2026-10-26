"""
config.py -- all file-system locations used by the release scripts.

Every path defaults to a location inside the release folder (the parent of this
``code/`` directory) and can be overridden with an environment variable:

    GENFIN_ROOT       project root (default: parent of code/)
    GENFIN_OUT        where experiment outputs are written (default: <root>/output)
    GENFIN_FIG_DIR    where build_master.py writes the final figure set (default: <root>/figures)
    GENFIN_YAHOO_DIR  folder with the Yahoo Finance statement CSVs (default: <root>/data/recent_data)
    GENFIN_SEC_FILE   Datavidence SEC XBRL sample file
                      (default: <root>/data/sec_datavidence/datavidence-financials-sample.csv.gz)

Values are plain strings so that they can be used with os.path.join.
"""
from __future__ import annotations

import os

_HERE = os.path.dirname(os.path.abspath(__file__))


def _env(name: str, default: str) -> str:
    v = os.environ.get(name, "").strip()
    return os.path.abspath(v) if v else os.path.abspath(default)


ROOT = _env("GENFIN_ROOT", os.path.join(_HERE, ".."))
OUT_MAIN = _env("GENFIN_OUT", os.path.join(ROOT, "output"))
FIG_DIR = _env("GENFIN_FIG_DIR", os.path.join(ROOT, "figures"))
REAL_DATA_DIR = _env("GENFIN_YAHOO_DIR", os.path.join(ROOT, "data", "recent_data"))
SEC_FILE = _env("GENFIN_SEC_FILE", os.path.join(ROOT, "data", "sec_datavidence",
                                                "datavidence-financials-sample.csv.gz"))

# Earlier result files that the consistency checks of E1, E3 and E4 compare against.
ORIGINAL_OUTPUTS = os.path.join(ROOT, "outputs_original")
BASELINE_DIR = os.path.join(ROOT, "generative", "baselines")

YAHOO_PANEL_FILE = os.path.join(REAL_DATA_DIR, "fundamentals_2022_2025.csv")


def have_yahoo_data() -> bool:
    return os.path.isdir(REAL_DATA_DIR) and any(
        f.endswith("_balance_sheet.csv") for f in os.listdir(REAL_DATA_DIR))


def have_sec_data() -> bool:
    return os.path.isfile(SEC_FILE)


if __name__ == "__main__":
    for k in ("ROOT", "OUT_MAIN", "FIG_DIR", "REAL_DATA_DIR", "SEC_FILE", "ORIGINAL_OUTPUTS", "BASELINE_DIR"):
        print(f"{k:17s} {globals()[k]}")
    print(f"{'yahoo data':17s} {'found' if have_yahoo_data() else 'missing'}")
    print(f"{'SEC file':17s} {'found' if have_sec_data() else 'missing'}")
