- `soup data mix --live` now validates supported text SFT configurations before training,
  isolates each candidate's run ID, tracker and output, and scores only that completed
  run's finite final-step validation loss. Training loss and stale runs cannot supply
  fallback scores; searches with no valid measurements exit nonzero without writing a
  recipe. The CLI and guide explicitly label the default offline mode as synthetic,
  with no model training or evaluation.
