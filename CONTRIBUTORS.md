# Contributors

Soup is built by its community. Thank you to everyone who has contributed code,
tests, docs, and ideas. ❤️

This list is maintained by hand alongside the GitHub
[contributors graph](https://github.com/MakazhanAlpamys/Soup/graphs/contributors).
Merged a PR and don't see yourself here? Open a PR adding your line — that counts too.

## Maintainer

- **Alpamys** ([@MakazhanAlpamys](https://github.com/MakazhanAlpamys)) — creator & lead maintainer

## Contributors

Listed by first contribution. PR numbers link the work.

- **Salil Mhatre** ([@Deadpool2000](https://github.com/Deadpool2000))
  - `soup version --json` for machine-readable CI output ([#6](https://github.com/MakazhanAlpamys/Soup/pull/6))
  - RAM + disk-space checks in `soup doctor` ([#7](https://github.com/MakazhanAlpamys/Soup/pull/7))
  - `soup runs clean` for smart checkpoint space management ([#9](https://github.com/MakazhanAlpamys/Soup/pull/9))
  - Official Docker support for easier onboarding ([#20](https://github.com/MakazhanAlpamys/Soup/pull/20))
  - `soup bench` — model speed + VRAM measurement ([#25](https://github.com/MakazhanAlpamys/Soup/pull/25))
  - `--prompts-file` option for `soup bench` ([#30](https://github.com/MakazhanAlpamys/Soup/pull/30))
  - Happy-path + CPU-warning tests for `soup bench` ([#31](https://github.com/MakazhanAlpamys/Soup/pull/31))
  - `soup cost` — cloud GPU training cost estimation ([#42](https://github.com/MakazhanAlpamys/Soup/pull/42))
  - `--nccl` flag for `soup doctor` multi-GPU bandwidth checks ([#178](https://github.com/MakazhanAlpamys/Soup/pull/178))
- **Chinmaya Sahu** ([@csking101](https://github.com/csking101))
  - DPO example config, sample data, and tests ([#48](https://github.com/MakazhanAlpamys/Soup/pull/48))
  - FP8 `rowwise` + `rowwise_with_gw_hp` scaling recipes ([#62](https://github.com/MakazhanAlpamys/Soup/pull/62))
- **Yixuan Xu** ([@mzl2233](https://github.com/mzl2233))
  - Guard diagnose-gate on distributed worker ranks ([#169](https://github.com/MakazhanAlpamys/Soup/pull/169))
- **dreamer0129** ([@dreamer0129](https://github.com/dreamer0129))
  - Rich-markup escape fix in legacy `soup adapters` commands ([#175](https://github.com/MakazhanAlpamys/Soup/pull/175), adopted in-tree as [#174](https://github.com/MakazhanAlpamys/Soup/issues/174))
- **Vivaan Dhawan** ([@VIVAAN-DHAWAN](https://github.com/VIVAAN-DHAWAN))
  - Reject pickle/zip streams renamed to `.safetensors` via magic-byte check ([#198](https://github.com/MakazhanAlpamys/Soup/pull/198))
- **Shivam** ([@shivam2931120](https://github.com/shivam2931120))
  - Tokenizer-aware repetition scoring for the echo-trap detector ([#242](https://github.com/MakazhanAlpamys/Soup/pull/242))
- **gittihub-jpg** ([@gittihub-jpg](https://github.com/gittihub-jpg))
  - Manifest-level dotted-path custom transforms for `soup build` ([#255](https://github.com/MakazhanAlpamys/Soup/pull/255))
  - `--energy` flag for `soup bom emit` — thread energy/CO₂ into the ML-BOM ([#256](https://github.com/MakazhanAlpamys/Soup/pull/256))
- **shatakshi-1404** ([@shatakshi-1404](https://github.com/shatakshi-1404))
  - Unit tests for the `warmup.py` auto-warmup-steps helper ([#274](https://github.com/MakazhanAlpamys/Soup/pull/274))

---

Want to join this list? See [CONTRIBUTING.md](CONTRIBUTING.md) — good first issues are
labelled in the [issue tracker](https://github.com/MakazhanAlpamys/Soup/labels/good%20first%20issue).
