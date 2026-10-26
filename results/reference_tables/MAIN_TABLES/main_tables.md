## Table 1

| Model | Original features, random 80/20 | Original features, grouped 5x5 | Strict, random 80/20 | Strict, grouped 5x5 mean (s.d.) | Strict, grouped 5x5 corrected 95% CI | Strict, temporal | Strict, unseen firms future years |
|---|---|---|---|---|---|---|---|
| Ridge | 0.695 | 0.664 | 0.250 | 0.180 (0.108) | [0.060, 0.301] | 0.235 | 0.245 |
| LinearReg | 0.700 | 0.666 | 0.249 | 0.180 (0.109) | [0.058, 0.301] | 0.231 | 0.243 |
| ElasticNet | 0.596 | 0.577 | 0.171 | 0.134 (0.045) | [0.084, 0.184] | 0.173 | 0.125 |
| CatBoost | 0.871 | 0.836 | 0.331 | 0.125 (0.146) | [-0.037, 0.288] | 0.326 | 0.165 |
| GradBoost | 0.887 | 0.852 | 0.303 | 0.108 (0.152) | [-0.060, 0.277] | 0.303 | 0.154 |
| SVR | 0.720 | 0.648 | 0.242 | 0.082 (0.147) | [-0.081, 0.245] | 0.202 | -0.030 |
| LightGBM | 0.895 | 0.858 | 0.316 | 0.082 (0.162) | [-0.098, 0.262] | 0.312 | 0.169 |
| XGBoost | 0.892 | 0.855 | 0.316 | 0.070 (0.156) | [-0.104, 0.244] | 0.318 | 0.155 |
| RandomForest | 0.863 | 0.817 | 0.311 | 0.045 (0.182) | [-0.157, 0.248] | 0.325 | 0.208 |
| Lasso | 0.516 | 0.494 | 0.046 | 0.030 (0.024) | [0.003, 0.057] | 0.054 | -0.049 |
| KNN | 0.529 | 0.430 | 0.111 | 0.029 (0.150) | [-0.138, 0.196] | 0.233 | -0.014 |
| MLP | 0.550 | 0.455 | 0.183 | 0.025 (0.150) | [-0.141, 0.191] | -1.552 | -1.952 |

## Table 2

| Generator | KS mean | W (ref. s.d.) | JSD (base 2) | Pearson MAD | CVR (any) | Logistic disc. AUC | DCR median |
|---|---|---|---|---|---|---|---|
| B1 noise baseline (original mechanism) | 0.136 ± 0.002 | 0.065 ± 0.003 | 0.084 ± 0.001 | 0.055 ± 0.003 | 0.656 ± 0.011 | 0.563 ± 0.006 | 0.688 ± 0.004 |
| B1-log noise baseline | 0.037 ± 0.003 | 0.036 ± 0.003 | 0.022 ± 0.000 | 0.033 ± 0.003 | 0.027 ± 0.005 | 0.551 ± 0.007 | 0.247 ± 0.000 |
| B2 Gaussian baseline | 0.101 ± 0.001 | 0.190 ± 0.004 | 0.086 ± 0.001 | 0.050 ± 0.002 | 0.022 ± 0.000 | 0.455 ± 0.011 | 0.533 ± 0.009 |
| CTGAN, 300 epochs | 0.210 ± 0.003 | 0.171 ± 0.025 | 0.074 ± 0.004 | 0.191 ± 0.001 | 1.000 ± 0.000 | 0.929 ± 0.016 | 0.828 ± 0.009 |
| CTGAN + projection, 300 epochs | 0.198 ± 0.011 | 0.179 ± 0.024 | 0.070 ± 0.011 | 0.182 ± 0.005 | 0.608 ± 0.116 | 0.912 ± 0.006 | 0.832 ± 0.023 |
| TVAE, 300 epochs | 0.092 ± 0.008 | 0.121 ± 0.013 | 0.032 ± 0.002 | 0.086 ± 0.008 | 0.999 ± 0.000 | 0.650 ± 0.002 | 0.480 ± 0.007 |
| TVAE + projection, 300 epochs | 0.098 ± 0.009 | 0.122 ± 0.014 | 0.035 ± 0.004 | 0.096 ± 0.010 | 0.347 ± 0.048 | 0.694 ± 0.014 | 0.497 ± 0.013 |
| TVAE, 600 epochs | 0.088 ± 0.007 | 0.119 ± 0.020 | 0.028 ± 0.001 | 0.082 ± 0.006 | 0.998 ± 0.000 | 0.614 ± 0.024 | 0.474 ± 0.003 |

## Table 3

| Sample | Specification | ROA coef. | SE | p | 95% CI | SE type |
|---|---|---|---|---|---|---|
| Reference synthetic panel (N = 4,440; 296 firms) | M1: Pooled OLS, original regressors, HC1 | 0.1388 | 0.0063 | <1e-15 | [0.126, 0.151] | HC1 |
| Reference synthetic panel (N = 4,440; 296 firms) | M2: Pooled OLS, original regressors, firm cluster | 0.1388 | 0.0141 | <1e-15 | [0.111, 0.167] | cluster(firm) |
| Reference synthetic panel (N = 4,440; 296 firms) | M3: Firm FE, original regressors, firm cluster | 0.0223 | 0.0043 | 1.9e-07 | [0.014, 0.031] | cluster(firm) |
| Reference synthetic panel (N = 4,440; 296 firms) | M4: Firm + year FE, original regressors, firm cluster | 0.0227 | 0.0045 | 4e-07 | [0.014, 0.031] | cluster(firm) |
| Reference synthetic panel (N = 4,440; 296 firms) | M5: Firm FE, original regressors, Driscoll-Kraay | 0.0223 | 0.0020 | <1e-15 | [0.018, 0.026] | Driscoll-Kraay (Bartlett, bw=2) |
| Reference synthetic panel (N = 4,440; 296 firms) | M6: Firm FE, strict regressors, firm cluster | 0.0019 | 0.0030 | 0.52 | [-0.004, 0.008] | cluster(firm) |
| Reference synthetic panel (N = 4,440; 296 firms) | M7: Pooled OLS, strict regressors, firm cluster | 0.0078 | 0.0122 | 0.53 | [-0.016, 0.032] | cluster(firm) |
| Reference synthetic panel (N = 4,440; 296 firms) | M8: Firm + year FE, strict regressors, firm cluster | 0.0031 | 0.0032 | 0.33 | [-0.003, 0.009] | cluster(firm) |
| Reference synthetic panel (N = 4,440; 296 firms) | aux: pooled, original minus interest_coverage, firm cluster | 0.0082 | 0.0121 | 0.5 | [-0.015, 0.032] | cluster(firm) |
| Reference synthetic panel (N = 4,440; 296 firms) | aux: pooled, original minus tobin_q, firm cluster | 0.1397 | 0.0143 | <1e-15 | [0.112, 0.168] | cluster(firm) |
| Reference synthetic panel (N = 4,440; 296 firms) | aux: pooled, roa + macro + sector only, firm cluster | 0.0077 | 0.0121 | 0.53 | [-0.016, 0.031] | cluster(firm) |
| Reference synthetic panel (N = 4,440; 296 firms) | aux: firm FE, original minus interest_coverage, firm cluster | 0.0019 | 0.0030 | 0.51 | [-0.004, 0.008] | cluster(firm) |
| Reference synthetic panel (N = 4,440; 296 firms) | aux: firm FE, roa + macro only, firm cluster | 0.0019 | 0.0030 | 0.52 | [-0.004, 0.008] | cluster(firm) |
| Real 46-firm sample (all) | Pooled OLS, year + sector dummies, spec A | 0.0619 | 0.2247 | 0.78 | [-0.378, 0.502] | cluster(firm) |
| Real 46-firm sample (all) | Firm + year FE, spec A | -0.3124 | 0.0942 | 0.0012 | [-0.499, -0.126] | cluster(firm) |
| Real 46-firm sample (all) | Pooled OLS, year + sector dummies, spec B | 0.0500 | 0.2277 | 0.83 | [-0.396, 0.496] | cluster(firm) |
| Real 46-firm sample (all) | Firm + year FE, spec B | -0.2933 | 0.1242 | 0.02 | [-0.539, -0.047] | cluster(firm) |
| Real 46-firm sample (excl_MCD) | Pooled OLS, year + sector dummies, spec A | 0.0190 | 0.2107 | 0.93 | [-0.394, 0.432] | cluster(firm) |
| Real 46-firm sample (excl_MCD) | Firm + year FE, spec A | -0.3111 | 0.0953 | 0.0014 | [-0.500, -0.123] | cluster(firm) |
| Real 46-firm sample (excl_MCD) | Pooled OLS, year + sector dummies, spec B | -0.0280 | 0.2033 | 0.89 | [-0.426, 0.370] | cluster(firm) |
| Real 46-firm sample (excl_MCD) | Firm + year FE, spec B | -0.2902 | 0.1241 | 0.021 | [-0.536, -0.045] | cluster(firm) |

