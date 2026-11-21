# RTX 5090 Laptop rule-derived retention

This is one capped-24-GiB, batch-1, sequence-2048 Mistral cell, not a physical 8-GiB GPU.

Rule: `uniform-base-flops-v1`. Native K=11; rule K=0, QKVO=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30], MLP=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30].

| Contrast | Median reduction | 95% interval | Family-16 interval |
|---|---:|---|---|
| mixed_vs_native | 8.166451% | [-5.59521735744494, 27.472056208829922] | [-15.799740862795174, 37.033010624233775] |
| mixed_vs_qkvo | 10.886924% | [-12.338090258203819, 28.846838834694516] | [-13.108285115172624, 37.29895804993961] |
| qkvo_vs_native | -1.401845% | [-12.058342317100065, 13.431081895144818] | [-16.562319278617164, 18.545262062697173] |

The earlier waiting-window waiver remains disclosed. A first native gate then failed the former 82 C execution guard, without a complete receipt or timing; its update count is unknown. After the user reboot, Standard/MSHybrid replaced Ultimate. The user explicitly authorized native thermal management during execution. This new operational amendment was frozen before models, is not published or owner-approved, and keeps the scientific sources, roster and statistical decisions unchanged. Results describe these amended operating conditions, including throttling; they are not a faithful execution of the originally published thermal conditions.

Primary stage passed: False. Confirmation passed: None.
Independent audit: 18 timing processes; 46 exact replays against two fresh-card reference updates. The failed prior verdicts remain unchanged.

The FLOP score prices Q/K/V/O and gate/up recomputation, not attention, dequantization, dispatch or memory transfers. It does not promise portable latency optimality. No empirical timing on this GPU selected or changed the rule.

Power limit is not exposed by this laptop driver. AC power and the current Performance scheme were checked; power draw, clocks and temperatures were sampled. No power setting was changed by the controller. Standard GPU mode was changed by the operator before this attempt. Stock thermal/power protection remained enabled. No service, driver, firmware or Soup production configuration was changed by this attempt.

WDDM desktop application identities and existing Chrome browser roots were frozen before execution. Chrome descendants of these roots were permitted with recorded ancestry; other unknown GPU processes were refused. Desktop GPU work is not isolated from these timings; quiet admission samples cannot establish continuous exclusive GPU use.

All process medians:

| Process | Median seconds | Allocated bytes | Reserved bytes |
|---|---:|---:|---:|
| block1-qkvo | 3.144839200 | 3543627264 | 3735027712 |
| block1-mixed | 2.789179150 | 7184283136 | 7434403840 |
| block1-native | 2.923953100 | 6547979776 | 7182745600 |
| block2-native | 2.975661450 | 6547979776 | 7182745600 |
| block2-mixed | 2.761109150 | 7184283136 | 7434403840 |
| block2-qkvo | 3.468500000 | 3543627264 | 3735027712 |
| block3-qkvo | 4.005838100 | 3543627264 | 3735027712 |
| block3-native | 4.369222000 | 6547979776 | 7182745600 |
| block3-mixed | 3.586646200 | 7184283136 | 7434403840 |
| block4-mixed | 4.199917750 | 7184283136 | 7434403840 |
| block4-native | 4.621523800 | 6547979776 | 7182745600 |
| block4-qkvo | 3.764450100 | 3543627264 | 3735027712 |
| block5-native | 3.455389900 | 6547979776 | 7182745600 |
| block5-qkvo | 3.537612250 | 3543627264 | 3735027712 |
| block5-mixed | 4.001332550 | 7184283136 | 7434403840 |
| block6-mixed | 1.882352150 | 7184283136 | 7434403840 |
| block6-qkvo | 3.002106650 | 3543627264 | 3735027712 |
| block6-native | 2.989426950 | 6547979776 | 7182745600 |

Thermal observations (temperature and flags: whole process; duration counters: owned CUDA context only; sampled flags are not exact duration fractions):

| Process | Peak GPU C | SW thermal us | HW thermal us | SW power cap us |
|---|---:|---:|---:|---:|
| gate-native-normal | 81 | 0 | 0 | 111076 |
| gate-qkvo-normal | 82 | 0 | 0 | 10750 |
| gate-mixed-normal | 84 | 0 | 0 | 110898 |
| gate-native-profile | 88 | 0 | 0 | 57876 |
| gate-qkvo-profile | 87 | 0 | 0 | 127605 |
| gate-mixed-profile | 86 | 0 | 0 | 116361 |
| block1-qkvo | 89 | 158661390 | 0 | 110838 |
| block1-mixed | 89 | 134390369 | 0 | 110401 |
| block1-native | 89 | 111984835 | 0 | 111020 |
| block2-native | 89 | 137923448 | 0 | 110890 |
| block2-mixed | 89 | 124566367 | 0 | 297382 |
| block2-qkvo | 89 | 212762944 | 0 | 171673 |
| block3-qkvo | 89 | 239274794 | 0 | 110488 |
| block3-native | 89 | 256202513 | 0 | 110783 |
| block3-mixed | 89 | 216430199 | 0 | 0 |
| block4-mixed | 90 | 257717877 | 0 | 3294349 |
| block4-native | 89 | 276023008 | 0 | 395307 |
| block4-qkvo | 89 | 233803587 | 0 | 111080 |
| block5-native | 89 | 209260696 | 0 | 110950 |
| block5-qkvo | 90 | 212523595 | 0 | 0 |
| block5-mixed | 89 | 227159446 | 0 | 106461 |
| block6-mixed | 88 | 45633431 | 0 | 110901 |
| block6-qkvo | 89 | 165434785 | 0 | 0 |
| block6-native | 90 | 176455617 | 0 | 110616 |

This entire campaign was repeated in Performance at the direct request of the operator. No Turbo receipts or timings were pooled or reused. The six Turbo gates and one timing, two interrupted Chrome attempts and earlier thermal failure remain preserved, with unknown extra partial updates. All scientific sources, layout rules, timing roster and thresholds were fixed unchanged before this separate run. The new native reference was generated in Performance.

The original frozen auditor assumed event counters were monotonic even before any owned CUDA context existed. Raw receipts disproved that assumption before model allocation. A separate audited helper preserves all readings, reports those pre-context decreases, and computes durations only over the owned CUDA-context lifetime. Decreases or disappearance inside that lifetime still fail. This bookkeeping correction was not published or owner-approved; the original auditor is preserved.

Operating changes within Performance: after six gates and two timings, the user briefly selected Turbo to cool during admission, then restored Performance; no worker started in that interrupted admission. After the third timing, the user requested removal of the remaining 75 C admission ceiling. The waiting controller and finalizer were replaced with no active model process, no lost updates and no repeated completed model. Nine complete processes retain their original bytes and original <=75 C admission checks. Only subsequent admission uses native thermal management without a software temperature ceiling; resource, identity, AC, power profile, three-sample idle admission and timing criteria remain unchanged. All temperature/throttling samples remain included. Cooling interventions and mixed admission regimes limit timing comparability; this is not uninterrupted or thermally standardized evidence. These local amendments were not published or approved by the owner.

A duplicate child-process admission guard was missed in the first no-ceiling helper. The fourth timing completed; the fifth process then rejected its inventory before importing torch or loading the model. That failed process made zero model updates and produced no timing. Its logs and original failure archive are preserved. A separate bootstrap now binds the same authorized admission function inside the child before executing the unchanged frozen harness. Ten completed processes were retained bit-exactly; zero completed models were repeated. The child binding is recorded and independently checked for every subsequent process. All prior operating-condition limitations still apply.

After fifteen complete Performance timings, a new Chrome GPU process descended from Playwright and a Claude task interrupted block6-mixed. The failed worker had no completed exactness or timing receipt; its partial model-update count is unknown. The operator subsequently agreed to pause that automation and close its browsers. Chrome and GPU process inventories were empty before this separate continuation. All six gates and fifteen complete timings were retained bit-exactly; only the incomplete block6-mixed and the two remaining timings resumed in their original order. No completed model was repeated. The long idle pause and cooling before the final block further limit thermal comparability. The original absolute deadline, sources, guard policy, layout, roster and statistical thresholds were unchanged. This amendment was not published or approved by the owner.
