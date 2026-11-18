# Fresh same-rule RTX 3090 qualification

This is one capped-24-GiB, batch-1, sequence-2048 Mistral cell, not a physical 8-GiB GPU.

Rule: `uniform-base-flops-v1`. Native K=11; rule K=0, QKVO=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30], MLP=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30].

| Contrast | Median reduction | 95% interval | Family-24 interval |
|---|---:|---|---|
| mixed_vs_native | 7.407784% | [6.999373985954742, 7.533636139268923] | [6.703648157686034, 7.58297803124208] |
| mixed_vs_qkvo | 10.717112% | [10.1360378581021, 10.84778564966512] | [10.134580324358408, 10.882873101834328] |
| qkvo_vs_native | -3.781183% | [-3.8471020196806265, -3.287310655569649] | [-3.855150641521904, -2.8426954843320873] |
| confirmation | 6.968553% | [6.520774388235448, 7.39832716744796] | [6.512996396342107, 7.41913489371679] |

Primary stage passed: True. Confirmation passed: True.
Independent audit: 30 timing processes; 70 exact replays against two fresh-card reference updates. The failed prior verdicts remain unchanged.

The FLOP score prices Q/K/V/O and gate/up recomputation, not attention, dequantization, dispatch or memory transfers. It does not promise portable latency optimality. No empirical timing on this GPU selected or changed the rule.

The second physical RTX 3090 retained its original 350 W setting. The earlier manual-layout result used GPU 0 at 420 W. These are paired comparisons on one card; no cross-card absolute speed or architecture-generalization claim is made.

Services stayed resident. Other GPUs were monitored for activity, Windows/Linux CI workers were refused, and inference slot task IDs remained fixed across all models. WDDM and WSL aggregate visibility do not prove full physical GPU exclusivity.

The user instructed immediate local execution without posting a new protocol. The complete criteria were frozen locally before any model, but the owner's requested prior-publication window was not followed. This limits process compliance and must accompany any later submission.

All process medians:

| Process | Median seconds | Allocated bytes | Reserved bytes |
|---|---:|---:|---:|
| block1-qkvo | 1.831853800 | 3543627264 | 3735027712 |
| block1-mixed | 1.635179600 | 7184283136 | 7434403840 |
| block1-native | 1.763854550 | 6547979776 | 7182745600 |
| block2-native | 1.754914300 | 6547979776 | 7182745600 |
| block2-mixed | 1.623571350 | 7184283136 | 7434403840 |
| block2-qkvo | 1.820406400 | 3543627264 | 3735027712 |
| block3-qkvo | 1.820136050 | 3543627264 | 3735027712 |
| block3-native | 1.753201350 | 6547979776 | 7182745600 |
| block3-mixed | 1.635672900 | 7184283136 | 7434403840 |
| block4-mixed | 1.635585300 | 7184283136 | 7434403840 |
| block4-native | 1.767466950 | 6547979776 | 7182745600 |
| block4-qkvo | 1.835320950 | 3543627264 | 3735027712 |
| block5-native | 1.766587100 | 6547979776 | 7182745600 |
| block5-qkvo | 1.832737150 | 3543627264 | 3735027712 |
| block5-mixed | 1.636673200 | 7184283136 | 7434403840 |
| block6-mixed | 1.635745200 | 7184283136 | 7434403840 |
| block6-qkvo | 1.820275550 | 3543627264 | 3735027712 |
| block6-native | 1.769960950 | 6547979776 | 7182745600 |
| confirm1-native | 1.752658950 | 6547979776 | 7182745600 |
| confirm1-mixed | 1.623510700 | 7184283136 | 7434403840 |
| confirm2-mixed | 1.638883150 | 7184283136 | 7434403840 |
| confirm2-native | 1.769422650 | 6547979776 | 7182745600 |
| confirm3-native | 1.753004250 | 6547979776 | 7182745600 |
| confirm3-mixed | 1.622946500 | 7184283136 | 7434403840 |
| confirm4-mixed | 1.638534100 | 7184283136 | 7434403840 |
| confirm4-native | 1.752978200 | 6547979776 | 7182745600 |
| confirm5-native | 1.753418750 | 6547979776 | 7182745600 |
| confirm5-mixed | 1.639218650 | 7184283136 | 7434403840 |
| confirm6-mixed | 1.637465650 | 7184283136 | 7434403840 |
| confirm6-native | 1.752582300 | 6547979776 | 7182745600 |
