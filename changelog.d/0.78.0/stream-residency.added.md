- Add opt-in `training.stream_resident_layers` for layer streaming (`training.stream_layers: true`):
  a count of centred decoder layers, or `auto`, that stay in private GPU slots for dense text SFT on
  the RAM tier or the async SSD source. `auto` measures the configured forward/backward peak with
  RNG and adapter state preserved, reserves future AdamW state, validates the candidate and falls
  back to ordinary streaming when no complete layer fits. Add opt-in `training.stream_ram_cache_mb`,
  a bounded RAM hotset of the most re-read layers on the SSD tier that is sized from the real
  allocation and the cgroup limits and excludes the GPU-resident layers. Both are off by default;
  residency is refused beside `activation_offloading` and a non-default `stream_recompute`. No
  speed guarantee (#PENDING by @kok-o, @nurseit14).
