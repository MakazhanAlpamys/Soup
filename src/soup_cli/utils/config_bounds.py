"""Config bounds shared by ``config/schema.py`` and the runtimes that enforce them.

A leaf: it imports nothing, so ``schema.py`` can take these values without
pulling the streaming and ship-verdict runtimes (and their ``rich`` subtree)
onto the ``soup version`` import path (#780). The runtime modules re-export
these names, so the schema bound and the runtime message stay one object.
"""

# --- layer streaming: buffers (utils/layer_stream.py) ----------------------
MIN_STREAM_BUFFERS = 2
MAX_STREAM_BUFFERS = 8
DEFAULT_STREAM_BUFFERS = 2

# --- layer streaming: read-ahead (utils/async_disk_source.py) --------------
MIN_STREAM_READ_AHEAD = 1
MAX_STREAM_READ_AHEAD = 8
DEFAULT_STREAM_READ_AHEAD = 2

# --- layer streaming: head + loss chunk (utils/stream_loss_chunk.py) --------
#: Below this the loop over the head costs more than the logits it saves: 16
#: tokens of a 128 256-token vocabulary are 29 MB at the shipped 14 B/element.
MIN_STREAM_LOSS_CHUNK = 16
MAX_STREAM_LOSS_CHUNK = 8192

# --- layer streaming: tasks ------------------------------------------------
#: Trainers that can run against a streamed base. DPO and IPO use a frozen
#: reference adapter on that base. KTO and BCO use the shared base for their
#: reference pass. ORPO and SimPO are reference-free.
SUPPORTED_STREAM_TASKS = ("sft", "dpo", "orpo", "simpo", "kto", "bco", "ipo")

#: Tasks PERMANENTLY excluded, not merely unimplemented. Generation rollouts
#: re-read every layer once per generated token, which destroys the whole
#: premise: streaming amortises one weight read over a training step, not over a
#: single decoded token (plan §3.2).
ROLLOUT_STREAM_TASKS = ("grpo", "ppo")

# --- ship verdict: noise floor (utils/ship_verdict.py) ---------------------
#: A floor needs a spread, and a spread needs at least two samples.
MIN_NOISE_FLOOR_RUNS = 2
#: Each run is a full pass over the base model; ten is already expensive.
MAX_NOISE_FLOOR_RUNS = 10
