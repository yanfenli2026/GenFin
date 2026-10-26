#!/usr/bin/env bash
# Regenerate every output of the paper in dependency order.
#
#   bash scripts/run_all.sh                 # full run (several hours on a CPU, see README)
#   PYTHON=/path/to/python bash scripts/run_all.sh
#   SKIP_SEC=1 bash scripts/run_all.sh      # skip E8 if the Datavidence file is not available
#   DOWNLOAD_YAHOO=1 bash scripts/run_all.sh  # fetch Yahoo Finance statements first (see README caveat)
#
# Paths can be redirected with GENFIN_OUT, GENFIN_FIG_DIR, GENFIN_YAHOO_DIR, GENFIN_SEC_FILE (code/config.py).
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
export MPLBACKEND=Agg PYTHONIOENCODING=utf-8
LOG_DIR="${GENFIN_LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

run() {
  local name="$1"; shift
  echo "=== [$(date +%H:%M:%S)] $name: $*"
  local t0=$SECONDS
  "$PY" "$@" 2>&1 | tee "$LOG_DIR/$name.log"
  echo "=== $name finished in $(( SECONDS - t0 )) s"
}

if [[ "${DOWNLOAD_YAHOO:-0}" == "1" ]]; then
  run yahoo_download data/download_recent_data.py
fi

if ! "$PY" -c "import sys; sys.path.insert(0,'code'); import config; sys.exit(0 if config.have_yahoo_data() else 1)"; then
  echo "ERROR: Yahoo Finance statement CSVs not found (see README, 'Data'). E3-E7 need them." >&2
  echo "       Run with DOWNLOAD_YAHOO=1 or set GENFIN_YAHOO_DIR." >&2
  exit 1
fi

export GENFIN_REQUIRE_REAL=1
run foundation code/test_common.py
run e1_benchmark code/e1_benchmark.py
run e2_deep_tabular code/e2_deep_tabular.py
run e3_shap code/e3_shap.py
run e4_generative code/e4_generative.py
run e5_optimization code/e5_optimization.py
run e6_econometrics code/e6_econometrics.py
run e7_real_data code/e7_real_data.py

if [[ "${SKIP_SEC:-0}" == "1" ]]; then
  echo "=== E8 skipped (SKIP_SEC=1); build_master.py will fail on the missing e8_sec tables"
elif "$PY" -c "import sys; sys.path.insert(0,'code'); import config; sys.exit(0 if config.have_sec_data() else 1)"; then
  run e8_sec_validation code_sec/e8_sec_validation.py
else
  echo "ERROR: Datavidence SEC sample not found (see README, 'Data'). Set GENFIN_SEC_FILE or SKIP_SEC=1." >&2
  exit 1
fi

run build_master code/build_master.py
run final_figures_main code/final_figures_main.py
run final_figures_si_a code/final_figures_si_a.py
run final_figures_si_b code/final_figures_si_b.py
echo "=== all done. Outputs: output/ (or \$GENFIN_OUT); figures: figures/ (or \$GENFIN_FIG_DIR)"
