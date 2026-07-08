# Phase-2 marginals: goodput vs concurrency

## gsm8k

| conc | baseline goodput (xbase) | W (w4a16) goodput (xbase) | K (fp8-kv) goodput (xbase) | S (eagle3) goodput (xbase) | emergent batch (mean/max, baseline) | tau (S) |
|---|---|---|---|---|---|---|
| 1 | 90 | 187 (2.07x) | 85 (0.94x) | 193 (2.13x) | 1.0 / 1 | 2.85 |
| 8 | 626 | 1065 (1.70x) | 593 (0.95x) | 1099 (1.76x) | 7.6 / 8 | 2.88 |
| 32 | 1591 | 2017 (1.27x) | 1566 (0.98x) | 1944 (1.22x) | 30.2 / 32 | 2.89 |
| 64 | 2237 | 2259 (1.01x) | 2291 (1.02x) | 2173 (0.97x) | 58.9 / 64 | 2.89 |

## humaneval

| conc | baseline goodput (xbase) | W (w4a16) goodput (xbase) | K (fp8-kv) goodput (xbase) | S (eagle3) goodput (xbase) | emergent batch (mean/max, baseline) | tau (S) |
|---|---|---|---|---|---|---|
| 1 | 92 | 196 (2.13x) | 86 (0.94x) | 290 (3.16x) | 1.0 / 1 | 4.09 |
| 8 | 688 | 1340 (1.95x) | 651 (0.95x) | 1874 (2.72x) | 7.7 / 8 | 4.08 |
| 32 | 2127 | 3331 (1.57x) | 2002 (0.94x) | 3924 (1.84x) | 29.6 / 32 | 4.08 |
| 64 | 3468 | 4103 (1.18x) | 3374 (0.97x) | 4737 (1.37x) | 58.7 / 64 | 4.07 |

## rag_shared_prefix

| conc | baseline goodput (xbase) | W (w4a16) goodput (xbase) | K (fp8-kv) goodput (xbase) | S (eagle3) goodput (xbase) | emergent batch (mean/max, baseline) | tau (S) |
|---|---|---|---|---|---|---|
| 1 | 90 | 188 (2.07x) | 85 (0.94x) | 172 (1.90x) | 1.0 / 1 | 2.52 |
| 8 | 627 | 1064 (1.70x) | 596 (0.95x) | 1013 (1.61x) | 7.6 / 8 | 2.58 |
| 32 | 1592 | 2012 (1.26x) | 1565 (0.98x) | 1795 (1.13x) | 30.1 / 32 | 2.57 |
| 64 | 2224 | 2234 (1.00x) | 2282 (1.03x) | 1998 (0.90x) | 59.7 / 64 | 2.54 |
