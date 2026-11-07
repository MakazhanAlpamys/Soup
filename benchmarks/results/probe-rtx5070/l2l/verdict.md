| gate | status | detail |
|---|---|---|
| G1 | pass | F1: 256 tensors + losses equal (deterministic); F2: 640 tensors + losses equal (deterministic) |
| G2 | pass | T_A(16) 73.3 vs 0.8 x T_B(16) 58.8 tok/s |
| G3 | no verdict | ('l2l', 'A', 1): V4: round spread 14.1% > 5% |
| G4 | no verdict | ('l2l', 'A', 1): V4: round spread 14.1% > 5% |

| row | ok | detail |
|---|---|---|
| V1 | True | direct_io=True |
| V2 | True | A: loads 159.0 vs 159.0, bytes 70.378 vs 70.378 GB; B: loads 159.0 vs 159.0, bytes 70.378 vs 70.378 GB |
| V3 | True | {'free_gb': 15.648956416, 'commit_avail_gb': 26.47990272, 'need_gb': 10.73741824, 'margin_gb': 4.0, 'gpu_used_before_build_mib': 0, 'gpu_pids': [13448], 'foreign_gpu_pids': [], 'ram_ok': True, 'gpu_ok': True, 'ok': True} |

| arm | valid | step s | tok/s | peak GB | why |
|---|---|---|---|---|---|
| plain/A/k1 | True | 17.396 | 29.4 | 4.509 |  |
| plain/B/k1 | True | 6.417 | 79.8 | 4.509 |  |
| l2l/A/k1 | False | 18.465 | 27.7 | 4.018 | V4: round spread 14.1% > 5% |
| l2l/B/k1 | True | 6.975 | 73.4 | 4.018 |  |
| l2l/A/k16 | True | 111.796 | 73.3 | 4.145 |  |
| l2l/B/k16 | True | 111.544 | 73.4 | 4.145 |  |
| l2l/A/k4 | False | 28.064 | 73.0 | 4.044 | V6: round 0 foreign reader SearchIndexer.exe 1.12 GB |
| l2l/B/k4 | True | 27.792 | 73.7 | 4.044 |  |
| l2l/C/k4 | True | 23.864 | 85.8 | 3.443 |  |
| l2l/D/k4 | True | 21.955 | 93.3 | 3.443 |  |
| l2l/A/k2 | True | 18.010 | 56.9 | 4.028 |  |
| l2l/A/k3 | False | 22.624 | 67.9 | 4.036 | V6: round 0 foreign reader SearchIndexer.exe 1.85 GB |

I2: {'dequant_share': 0.21001882973534314, 'c4_over_a4': None}

I3: {'overhead': 1.0864772852203788}

I4: {'ratio': 0.998344384854415, 'free': True, 'direct': True}

I6: {'b': 1.0870835247708035}

**Verdict: NO VERDICT: G3, G4**
