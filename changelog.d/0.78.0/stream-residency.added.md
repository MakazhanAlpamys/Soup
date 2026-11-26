- Add opt-in `training.stream_resident_layers` for layer streaming (`training.stream_layers: true`):
  a count of centred decoder layers, or `auto`, that stay in private GPU slots for dense text SFT on
  the RAM tier. `auto` measures the configured forward/backward peak with RNG and adapter state
  preserved, reserves future AdamW state, validates the candidate and falls back to ordinary
  streaming when no complete layer fits. Off by default; refused beside `activation_offloading`
  and a non-default `stream_recompute`. No speed guarantee (#PENDING by @kok-o, @nurseit14).
