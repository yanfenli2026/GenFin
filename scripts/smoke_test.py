"""
smoke_test.py -- quick check that the release installs and runs (no pytest needed).

    python scripts/smoke_test.py            # about 30 s on a laptop CPU

What it does
  1. byte-compiles every script in code/, code_sec/, data/ and scripts/;
  2. imports config and common and prints the resolved paths;
  3. runs code/test_common.py in a subprocess with GENFIN_OUT pointing to a temporary
     folder, so nothing in output/ is overwritten;
  4. compares three of the files test_common.py writes with the reference copies in
     results/reference_tables/SUPP_TABLES (S01 leakage audit, S03 DGP parameters,
     S06 variance decomposition of the synthetic debt ratio).

Exit code 0 means every step passed.
"""
from __future__ import annotations

import os
import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REL = Path(__file__).resolve().parents[1]
CODE = REL / "code"
REF = REL / "results" / "reference_tables" / "SUPP_TABLES"


def step(msg):
    print(f"\n== {msg}", flush=True)


def main() -> int:
    step("1/4 byte-compile")
    files = sorted(CODE.glob("*.py")) + sorted((REL / "code_sec").glob("*.py")) + sorted((REL / "data").glob("*.py")) + sorted((REL / "scripts").glob("*.py"))
    for f in files:
        py_compile.compile(str(f), doraise=True)
    print(f"compiled {len(files)} files")

    step("2/4 configuration")
    sys.path.insert(0, str(CODE))
    import config  # noqa: E402
    import common  # noqa: E402
    for k in ("ROOT", "OUT_MAIN", "REAL_DATA_DIR", "SEC_FILE"):
        print(f"{k:14s} {getattr(config, k)}")
    print("Yahoo data   :", "found" if config.have_yahoo_data() else "missing (section 5 of test_common will be skipped)")
    print("SEC file     :", "found" if config.have_sec_data() else "missing (needed only for E8)")
    df = common.generate_panel()
    assert df.shape[0] == 4440, df.shape
    print("generate_panel(): ", df.shape, "rows x cols")

    step("3/4 code/test_common.py (outputs to a temporary folder)")
    with tempfile.TemporaryDirectory(prefix="genfin_smoke_") as tmp:
        env = dict(os.environ, GENFIN_OUT=tmp, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
        r = subprocess.run([sys.executable, str(CODE / "test_common.py")], cwd=str(REL), env=env,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        print(r.stdout[-3000:])
        if r.returncode != 0 or "ALL TESTS PASSED (incl. section 7)" not in r.stdout:
            print(r.stderr[-3000:])
            print("FAIL: test_common.py did not finish")
            return 1

        step("4/4 compare with reference tables")
        found = Path(tmp) / "foundation"
        checks = [("feature_audit.csv", "S01_feature_leakage_audit.csv"),
                  ("dgp_parameters.csv", "S03_dgp_parameters.csv"),
                  ("synthetic_debt_ratio_variance_decomposition.csv", "S06_synthetic_variance_decomposition.csv")]
        ok = True
        for new, ref in checks:
            a, b = pd.read_csv(found / new), pd.read_csv(REF / ref)
            same = a.shape == b.shape and list(a.columns) == list(b.columns)
            if same:
                for c in a.columns:
                    if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
                        same &= bool(np.allclose(a[c].to_numpy(float), b[c].to_numpy(float),
                                                 rtol=1e-9, atol=1e-12, equal_nan=True))
                    else:
                        same &= a[c].astype(str).tolist() == b[c].astype(str).tolist()
            print(f"{'OK  ' if same else 'DIFF'} {new} vs {ref}")
            ok &= same
        if not ok:
            print("FAIL: regenerated foundation tables differ from the reference copies")
            return 1
    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
