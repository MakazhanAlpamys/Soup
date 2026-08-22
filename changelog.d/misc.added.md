**Layer streaming now accepts Qwen3.5 MoE text checkpoints whose decoder
  layers do not expose exactly the same weight keys in every block (by
  @Shutaru in #426).** The sharder now records per-layer shard headers instead of
  deriving the pool from layer 0 alone, while still refusing divergent storage
  layouts for any shared key and rebuilding NF4 `Params4bit` views from
  validated per-weight metadata. `qwen3_5_moe` / `qwen3_5_moe_text` route
  through the existing qwen3 streamer. A heterogeneous toy MoE is bit-exact
  streamed versus resident on CPU; live validation was also run on
  `Qwen/Qwen3.5-35B-A3B` with layer streaming, NF4, MoE LoRA target resolution
  and a 3072-token SFT dataset. `stream_layers` now refuses
  `moe_expert_quant`, which is applied only by the resident setup path and was
  otherwise silently ignored.
