# Core 2^3 factorial: log-space effects on goodput_tok_s

Effect columns: log-effect (multiplicative factor) [min..max across complete repeats]; '~0?' = spread straddles zero. Gap > 0 = sub-additive interference (see analysis/factorial.py docstring).

## gsm8k @ concurrency 1

| effect | estimate |
|---|---|
| W | +0.440 (x1.55) [+0.437..+0.441] |
| K | -0.354 (x0.70) [-0.361..-0.350] |
| S | +0.310 (x1.36) [+0.296..+0.319] |
| WK | -0.093 (x0.91) [-0.094..-0.092] |
| WS | -0.253 (x0.78) [-0.257..-0.246] |
| KS | -0.258 (x0.77) [-0.266..-0.253] |
| WKS | -0.061 (x0.94) [-0.062..-0.059] |

**Interference gap**: naive x4.15 vs measured x1.40 -> gap +1.087 log (naive overestimates by x2.97) [+1.069..+1.106 across repeats]
Complete repeats: [0, 1, 2]

## gsm8k @ concurrency 8

| effect | estimate |
|---|---|
| W | +0.265 (x1.30) [+0.259..+0.275] |
| K | -0.234 (x0.79) [-0.238..-0.229] |
| S | +0.156 (x1.17) [+0.153..+0.161] |
| WK | -0.020 (x0.98) [-0.025..-0.014] |
| WS | -0.245 (x0.78) [-0.249..-0.240] |
| KS | -0.162 (x0.85) [-0.167..-0.156] |
| WKS | -0.002 (x1.00) [-0.006..+0.002] ~0? |

**Interference gap**: naive x2.82 vs measured x1.20 -> gap +0.852 log (naive overestimates by x2.35) [+0.846..+0.862 across repeats]
Complete repeats: [0, 1, 2]

## gsm8k @ concurrency 32

| effect | estimate |
|---|---|
| W | +0.027 (x1.03) [+0.024..+0.031] |
| K | -0.101 (x0.90) [-0.106..-0.095] |
| S | -0.097 (x0.91) [-0.101..-0.094] |
| WK | +0.002 (x1.00) [-0.001..+0.006] ~0? |
| WS | -0.208 (x0.81) [-0.209..-0.205] |
| KS | -0.084 (x0.92) [-0.089..-0.079] |
| WKS | +0.003 (x1.00) [+0.001..+0.007] |

**Interference gap**: naive x1.52 vs measured x0.85 -> gap +0.585 log (naive overestimates by x1.80) [+0.580..+0.591 across repeats]
Complete repeats: [0, 1, 2]

## gsm8k @ concurrency 64

| effect | estimate |
|---|---|
| W | -0.109 (x0.90) [-0.111..-0.106] |
| K | -0.032 (x0.97) [-0.038..-0.024] |
| S | -0.205 (x0.81) [-0.208..-0.201] |
| WK | +0.001 (x1.00) [-0.001..+0.005] ~0? |
| WS | -0.113 (x0.89) [-0.117..-0.111] |
| KS | -0.053 (x0.95) [-0.057..-0.047] |
| WKS | +0.005 (x1.00) [+0.001..+0.009] |

**Interference gap**: naive x1.00 vs measured x0.71 -> gap +0.338 log (naive overestimates by x1.40) [+0.322..+0.347 across repeats]
Complete repeats: [0, 1, 2]

## humaneval @ concurrency 1

| effect | estimate |
|---|---|
| W | +0.527 (x1.69) [+0.525..+0.530] |
| K | -0.383 (x0.68) [-0.386..-0.380] |
| S | +0.746 (x2.11) [+0.739..+0.756] |
| WK | -0.101 (x0.90) [-0.102..-0.099] |
| WS | -0.190 (x0.83) [-0.193..-0.188] |
| KS | -0.281 (x0.76) [-0.284..-0.278] |
| WKS | -0.065 (x0.94) [-0.065..-0.064] |

**Interference gap**: naive x6.29 vs measured x2.28 -> gap +1.014 log (naive overestimates by x2.76) [+1.006..+1.025 across repeats]
Complete repeats: [0, 1, 2]

## humaneval @ concurrency 8

| effect | estimate |
|---|---|
| W | +0.419 (x1.52) [+0.410..+0.426] |
| K | -0.285 (x0.75) [-0.288..-0.278] |
| S | +0.586 (x1.80) [+0.584..+0.589] |
| WK | -0.032 (x0.97) [-0.037..-0.028] |
| WS | -0.220 (x0.80) [-0.226..-0.214] |
| KS | -0.201 (x0.82) [-0.206..-0.197] |
| WKS | -0.008 (x0.99) [-0.015..-0.002] |

**Interference gap**: naive x4.97 vs measured x2.04 -> gap +0.890 log (naive overestimates by x2.44) [+0.887..+0.893 across repeats]
Complete repeats: [0, 1, 2]

## humaneval @ concurrency 32

| effect | estimate |
|---|---|
| W | +0.190 (x1.21) [+0.181..+0.199] |
| K | -0.168 (x0.85) [-0.173..-0.162] |
| S | +0.260 (x1.30) [+0.257..+0.262] |
| WK | -0.006 (x0.99) [-0.013..+0.003] ~0? |
| WS | -0.248 (x0.78) [-0.254..-0.240] |
| KS | -0.097 (x0.91) [-0.103..-0.093] |
| WKS | +0.005 (x1.00) [-0.002..+0.013] ~0? |

**Interference gap**: naive x2.70 vs measured x1.33 -> gap +0.709 log (naive overestimates by x2.03) [+0.703..+0.719 across repeats]
Complete repeats: [0, 1, 2]

## humaneval @ concurrency 64

| effect | estimate |
|---|---|
| W | +0.023 (x1.02) [+0.019..+0.029] |
| K | -0.102 (x0.90) [-0.106..-0.100] |
| S | +0.092 (x1.10) [+0.090..+0.093] |
| WK | +0.000 (x1.00) [-0.003..+0.007] ~0? |
| WS | -0.139 (x0.87) [-0.142..-0.135] |
| KS | -0.070 (x0.93) [-0.075..-0.068] |
| WKS | +0.005 (x1.00) [+0.001..+0.011] |

**Interference gap**: naive x1.56 vs measured x1.02 -> gap +0.428 log (naive overestimates by x1.53) [+0.425..+0.432 across repeats]
Complete repeats: [0, 1, 2]

## rag_shared_prefix @ concurrency 1

| effect | estimate |
|---|---|
| W | +0.497 (x1.64) [+0.490..+0.505] |
| K | -0.357 (x0.70) [-0.366..-0.347] |
| S | +0.241 (x1.27) [+0.236..+0.245] |
| WK | -0.093 (x0.91) [-0.102..-0.087] |
| WS | -0.196 (x0.82) [-0.198..-0.192] |
| KS | -0.261 (x0.77) [-0.271..-0.252] |
| WKS | -0.060 (x0.94) [-0.070..-0.054] |

**Interference gap**: naive x3.67 vs measured x1.38 -> gap +0.979 log (naive overestimates by x2.66) [+0.965..+0.990 across repeats]
Complete repeats: [0, 1, 2]

## rag_shared_prefix @ concurrency 8

| effect | estimate |
|---|---|
| W | +0.310 (x1.36) [+0.302..+0.319] |
| K | -0.233 (x0.79) [-0.241..-0.223] |
| S | +0.121 (x1.13) [+0.116..+0.128] |
| WK | -0.022 (x0.98) [-0.029..-0.013] |
| WS | -0.197 (x0.82) [-0.204..-0.192] |
| KS | -0.162 (x0.85) [-0.168..-0.152] |
| WKS | -0.004 (x1.00) [-0.012..+0.003] ~0? |

**Interference gap**: naive x2.58 vs measured x1.21 -> gap +0.754 log (naive overestimates by x2.12) [+0.739..+0.773 across repeats]
Complete repeats: [0, 1, 2]

## rag_shared_prefix @ concurrency 32

| effect | estimate |
|---|---|
| W | +0.065 (x1.07) [+0.061..+0.069] |
| K | -0.100 (x0.91) [-0.102..-0.097] |
| S | -0.138 (x0.87) [-0.140..-0.135] |
| WK | +0.005 (x1.00) [+0.001..+0.012] |
| WS | -0.168 (x0.85) [-0.171..-0.163] |
| KS | -0.081 (x0.92) [-0.084..-0.078] |
| WKS | +0.007 (x1.01) [+0.003..+0.013] |

**Interference gap**: naive x1.40 vs measured x0.85 -> gap +0.502 log (naive overestimates by x1.65) [+0.496..+0.505 across repeats]
Complete repeats: [0, 1, 2]

## rag_shared_prefix @ concurrency 64

| effect | estimate |
|---|---|
| W | -0.078 (x0.92) [-0.079..-0.078] |
| K | -0.027 (x0.97) [-0.030..-0.023] |
| S | -0.244 (x0.78) [-0.246..-0.242] |
| WK | +0.002 (x1.00) [+0.001..+0.004] |
| WS | -0.077 (x0.93) [-0.078..-0.077] |
| KS | -0.049 (x0.95) [-0.051..-0.048] |
| WKS | +0.005 (x1.01) [+0.003..+0.010] |

**Interference gap**: naive x0.92 vs measured x0.71 -> gap +0.260 log (naive overestimates by x1.30) [+0.255..+0.262 across repeats]
Complete repeats: [0, 1, 2]
