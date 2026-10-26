# Security notes

This repository contains research code only. It holds no credentials, API keys or personal data,
and it makes no network calls except `data/download_recent_data.py`, which uses the `yfinance`
package to fetch public annual statements from Yahoo Finance.

## Pinned dependencies

`requirements.txt` and `requirements-freeze.txt` pin the exact versions used to produce the
paper, so that the results can be reproduced. Some of these versions have known advisories,
for example in `torch` 2.12.0 (`torch.jit.script`), `pillow`, `urllib3` and `setuptools`. The
release scripts do not call `torch.jit` and do not process untrusted images or URLs. Install the
environment in a dedicated virtual environment, and use newer versions for any other purpose.

## Local caches

`code/e2_deep_tabular.py` caches finished model fits as pickle files under
`output/e2_deep_tabular/cache/`, and `code/e4_generative.py` caches CTGAN/TVAE samples
under `output/e4_generative/deep_cache/`; both reload them on a re-run. Pickle files can execute
code when loaded, so only reuse caches that this code wrote on your own machine; delete these
folders before using a cache from any other source.

## Reporting

Please report problems through the issue tracker of this repository.
