# Computing environment used for the paper

| Item | Value |
|---|---|
| Python | 3.13.12 (CPython, 64-bit) |
| OS | Windows 11 (64-bit) |
| CPU | x86-64, 32 logical cores, about 32 GB RAM |
| GPU | none used (PyTorch CPU build, torch 2.12.0+cpu) |
| Packages | `requirements.txt` (direct, pinned); `requirements-freeze.txt` (full snapshot) |

Setup (any OS):

    python -m venv .venv
    # Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
    pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt
    python scripts/smoke_test.py

Notes
- All random seeds are fixed in the scripts (seed 42 unless stated in the experiment's RESULTS.md).
  The synthetic-panel parts reproduce bit-for-bit on the same package versions; tree models,
  PyTorch models and CTGAN/TVAE can differ in late decimals across CPUs, BLAS builds, thread
  counts and operating systems.
- Several scripts run work in parallel (`n_jobs = 4`; E2 `--workers 5`). Reduce these on machines
  with fewer cores; results do not depend on the number of workers, only run time does.
- Figures are written with the non-interactive Matplotlib backend (`MPLBACKEND=Agg` is set by the
  run scripts).
- The code was only run on Windows. Paths are built with `os.path`, so Linux/macOS should work,
  but this has not been tested.
