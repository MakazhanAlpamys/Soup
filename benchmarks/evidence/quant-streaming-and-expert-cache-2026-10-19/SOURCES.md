# Evidence sources

| File | Source path | Source commit | Size (bytes) | SHA-256 | Copied | Notes |
| --- | --- | --- | ---: | --- | --- | --- |
| d1-quant-quality-result-20261019.md | benchmarks/d1-quant-quality-result-20261019.md | 06bb5d1c1cf2fa4cfad4ba877e9b4b6733da1b9a | 6651 | 0b748980f5f1b15223840bb4c8996a6dc1de28aca62762b76c8f62a5dc35e195 | no | Companion report for quality loss, dataset and setup. Gives rounded loss values. Contains billing and internal labels. |
| d1-vm-f1-result-20261019.md | benchmarks/d1-vm-f1-result-20261019.md | 846a0dda50735154078a96b7ef9c24838f3b3a61 | 8892 | 73bac106ff4ab2b434e04873e3d8c78fa8d4b00aa0251e619497144fc6a70bc9 | no | Companion report for synthetic streaming timing, bytes and block counts. Gives rounded timings. Contains billing and internal labels. |
| d1-real7b-result-20261019.md | benchmarks/d1-real7b-result-20261019.md | 820bd25eabe51d01005a0d86b8fb14936893349a | 8163 | ff43e03bb904f20a7e602d8a7542a0c33ef443d52f865e7ce531a9176cc3bde1 | no | Companion report for Mistral streaming timing, bytes and validity. Gives rounded timings. Contains billing and internal labels. |
| cpu-cache-policy-report.md | benchmarks/expert-cache-policy-20261025/README.md | f6413e528d189410b30d8841158aed85cbec2b7d | 5375 | 4102a6f69078a06ae3630f7c639f445f8024ba6066d6df8dd7fd0dfa531da73a | no | CPU host and replay method, modeled reads, cloned payload and operation times. Contains a private link. |
| cpu-evidence.zip | benchmarks/expert-cache-policy-20261025/cpu-evidence.zip | f6413e528d189410b30d8841158aed85cbec2b7d | 2003650 | 6f3d051270a508a34e14ca7b8ddee28ffd4e1f2d01db6343155a5d031c04a134 | no | Raw archive over 1 MiB. Its CPU timing JSON files supply modeled reads, clone bytes and wall times. |
| kanana-preparation.md | benchmarks/kanana-30b-cache-20261026/README.md | f6413e528d189410b30d8841158aed85cbec2b7d | 4988 | b8d07fc0ad05655ed1a706fd2f86733c3745273708a413c8fe9a4f59b89fc544 | no | Supplies the planned recipe, header checks and tiny CPU controls. Contains billing and private workflow details. |
| gpu-cache-policy-summary.json | benchmarks/expert-cache-policy-20261025/gpu-results-summary.json | 4bfccb15a3eb646acf0118b353386b2773c83138 | 40917 | f4742a0aa53205a01ff2092567da52734ab2e30d4580d27c0155836204bcabf6 | no | Supplies the four GPU arms, paired reductions, numerical gate and runtime details. Contains billing figures. |
| gpu-evidence.zip | benchmarks/expert-cache-policy-20261025/gpu-evidence.zip | 4bfccb15a3eb646acf0118b353386b2773c83138 | 3219165 | 94972747e1bfda75b8818982d495db16a83176fc60d07bed14ad15ba07dbd63a | no | Raw archive over 1 MiB. Supplies exact timing and parity receipts and the prepared recipe. |
| earlier-gpu-cache-summary.json | benchmarks/two-read-expert-cache-20261025/gpu-results-summary.json | 0beacac93b3184c2b04584c9ab9fc9e71d267852 | 35057 | efb5d52589177598d47be3d06a27b4b9a5d8de953cd2ac5cc8e2fceaa4351c5b | no | Supplies the earlier paired LRU result. Contains billing figures. |

The tree at source commit `33a68aac8c6fec6274866aee836850a4ebb473d4` identifies these three October audit blobs. The local clone lacks their contents, so their byte counts, SHA-256 digests and full-precision figures are unverified here. The companion reports above give the rounded figures.

- `followups/quant-quality-20261019/audit.json` (Git blob `6b63e2b1228e27ce1f1aca3ebadd239dd76f9013`)
- `followups/vm-f1-20261019/audit.json` (Git blob `f897f8b53389bae9da408933ed0f1b099491809c`)
- `followups/real7b-20261019/audit.json` (Git blob `93b90f3428845e0b3edaf6a5c366dd0012bfb602`)
