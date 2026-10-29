- `training.activation_offloading` now reaches the VRAM estimate: the `soup train`
  pre-flight, `soup profile` (text and `--json`), `soup plan` and the layer-streaming
  plan panel say, when it is set, that the activation term is still charged as if every
  saved tensor stayed on the card, since what `cpu` or `disk` offloading moves off it
  is not modelled. The estimate and the pre-flight's refuse or warn decision are
  unchanged (#2327 by @ybai08 in #PRNUMBER).
