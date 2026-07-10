# Core 2^3 factorial: log-space effects on goodput_tok_s

Effect columns: log-effect (multiplicative factor) [min..max across complete repeats]; '~0?' = spread straddles zero. Gap > 0 = sub-additive interference (see analysis/factorial.py docstring).

## gsm8k @ concurrency 1

| effect | estimate |
|---|---|
| W | +0.441 (x1.55) |
| K | -0.361 (x0.70) |
| S | +0.296 (x1.35) |
| WK | -0.093 (x0.91) |
| WS | -0.256 (x0.77) |
| KS | -0.266 (x0.77) |
| WKS | -0.061 (x0.94) |

**Interference gap**: naive x4.15 vs measured x1.37 -> gap +1.106 log (naive overestimates by x3.02)
Complete repeats: [0]

## gsm8k @ concurrency 8

| effect | estimate |
|---|---|
| W | +0.275 (x1.32) |
| K | -0.238 (x0.79) |
| S | +0.154 (x1.17) |
| WK | -0.014 (x0.99) |
| WS | -0.240 (x0.79) |
| KS | -0.167 (x0.85) |
| WKS | +0.002 (x1.00) |

**Interference gap**: naive x2.83 vs measured x1.21 -> gap +0.846 log (naive overestimates by x2.33)
Complete repeats: [0]

## gsm8k @ concurrency 32

| effect | estimate |
|---|---|
| W | +0.031 (x1.03) |
| K | -0.106 (x0.90) |
| S | -0.101 (x0.90) |
| WK | +0.006 (x1.01) |
| WS | -0.205 (x0.81) |
| KS | -0.089 (x0.91) |
| WKS | +0.007 (x1.01) |

**Interference gap**: naive x1.53 vs measured x0.84 -> gap +0.591 log (naive overestimates by x1.81)
Complete repeats: [0]

## gsm8k @ concurrency 64

| effect | estimate |
|---|---|
| W | -0.106 (x0.90) |
| K | -0.038 (x0.96) |
| S | -0.208 (x0.81) |
| WK | +0.005 (x1.01) |
| WS | -0.112 (x0.89) |
| KS | -0.057 (x0.94) |
| WKS | +0.009 (x1.01) |

**Interference gap**: naive x1.01 vs measured x0.71 -> gap +0.347 log (naive overestimates by x1.41)
Complete repeats: [0]

## humaneval @ concurrency 1

| effect | estimate |
|---|---|
| W | +0.528 (x1.70) |
| K | -0.386 (x0.68) |
| S | +0.739 (x2.09) |
| WK | -0.101 (x0.90) |
| WS | -0.193 (x0.82) |
| KS | -0.284 (x0.75) |
| WKS | -0.065 (x0.94) |

**Interference gap**: naive x6.31 vs measured x2.26 -> gap +1.025 log (naive overestimates by x2.79)
Complete repeats: [0]

## humaneval @ concurrency 8

| effect | estimate |
|---|---|
| W | +0.426 (x1.53) |
| K | -0.287 (x0.75) |
| S | +0.584 (x1.79) |
| WK | -0.028 (x0.97) |
| WS | -0.214 (x0.81) |
| KS | -0.206 (x0.81) |
| WKS | -0.002 (x1.00) |

**Interference gap**: naive x5.03 vs measured x2.06 -> gap +0.893 log (naive overestimates by x2.44)
Complete repeats: [0]

## humaneval @ concurrency 32

| effect | estimate |
|---|---|
| W | +0.199 (x1.22) |
| K | -0.173 (x0.84) |
| S | +0.257 (x1.29) |
| WK | +0.003 (x1.00) |
| WS | -0.240 (x0.79) |
| KS | -0.103 (x0.90) |
| WKS | +0.013 (x1.01) |

**Interference gap**: naive x2.72 vs measured x1.34 -> gap +0.705 log (naive overestimates by x2.02)
Complete repeats: [0]

## humaneval @ concurrency 64

| effect | estimate |
|---|---|
| W | +0.029 (x1.03) |
| K | -0.106 (x0.90) |
| S | +0.091 (x1.10) |
| WK | +0.007 (x1.01) |
| WS | -0.135 (x0.87) |
| KS | -0.075 (x0.93) |
| WKS | +0.011 (x1.01) |

**Interference gap**: naive x1.57 vs measured x1.02 -> gap +0.428 log (naive overestimates by x1.53)
Complete repeats: [0]

## rag_shared_prefix @ concurrency 1

| effect | estimate |
|---|---|
| W | +0.505 (x1.66) |
| K | -0.366 (x0.69) |
| S | +0.236 (x1.27) |
| WK | -0.087 (x0.92) |
| WS | -0.192 (x0.83) |
| KS | -0.271 (x0.76) |
| WKS | -0.054 (x0.95) |

**Interference gap**: naive x3.71 vs measured x1.38 -> gap +0.990 log (naive overestimates by x2.69)
Complete repeats: [0]

## rag_shared_prefix @ concurrency 32

| effect | estimate |
|---|---|
| W | +0.069 (x1.07) |
| K | -0.102 (x0.90) |
| S | -0.140 (x0.87) |
| WK | +0.012 (x1.01) |
| WS | -0.163 (x0.85) |
| KS | -0.084 (x0.92) |
| WKS | +0.013 (x1.01) |

**Interference gap**: naive x1.40 vs measured x0.85 -> gap +0.496 log (naive overestimates by x1.64)
Complete repeats: [0]

## rag_shared_prefix @ concurrency 64

| effect | estimate |
|---|---|
| W | -0.079 (x0.92) |
| K | -0.029 (x0.97) |
| S | -0.242 (x0.79) |
| WK | +0.004 (x1.00) |
| WS | -0.077 (x0.93) |
| KS | -0.048 (x0.95) |
| WKS | +0.010 (x1.01) |

**Interference gap**: naive x0.93 vs measured x0.71 -> gap +0.262 log (naive overestimates by x1.30)
Complete repeats: [0]

## Incomplete cubes (skipped)
- rag_shared_prefix @ conc=8
