"""Fresh-process real native continuation worker; no model/network downloads."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--stop", type=int)
    parser.add_argument("--stop-micro", type=int)
    parser.add_argument("--cli", action="store_true")
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    import random

    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten

    from soup_cli.config.schema import SoupConfig
    from soup_cli.trainer.mlx_checkpoint import FullStateSession, preserve_random
    from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper

    cfg = SoupConfig(**json.loads(Path(args.config).read_text()))
    trace, batches, updates, frozen = [], [], [], {}
    original_apply = MLXSFTTrainerWrapper._apply_lora
    original_train = MLXSFTTrainerWrapper.train
    original_iterate = FullStateSession.iterate
    original_save = FullStateSession.save
    original_prepare = FullStateSession.prepare
    captured = {}

    def prepare(session):
        original_prepare(session)
        captured["prepared"] = {
            k: mx.array(v) for k, v in tree_flatten(session.model.trainable_parameters())
        }
        mx.eval(captured["prepared"])
        if session.bundle is None:
            # Exercise actual advanced host RNG states even though this native
            # sampler owns a separate NumPy stream. Restoring seeds must fail.
            np.random.random(7)
            for _ in range(5):
                random.random()
            random.gauss(0, 1)  # Keep the Gaussian cache as part of saved state.

    def apply(wrapper, model):
        original_apply(wrapper, model)
        trainable = dict(tree_flatten(model.trainable_parameters()))
        frozen.update(
            {k: mx.array(v) for k, v in tree_flatten(model.parameters()) if k not in trainable}
        )
        mx.eval(frozen)

    def train(wrapper, *positional, **kwargs):
        captured["model"] = wrapper.model
        result = original_train(wrapper, *positional, **kwargs)
        captured["result"] = result
        return result

    def iterate(session, **kwargs):
        count = session.start_micro
        for batch in original_iterate(session, **kwargs):
            if kwargs.get("loop"):
                if args.stop_micro and count == args.stop_micro:
                    raise KeyboardInterrupt("deliberate mid-accumulation interruption")
                count += 1
                batches.append({"microbatch": count, "batch": [x.tolist() for x in batch]})
            yield batch

    def save(session, microbatch):
        with preserve_random():
            updates.append(
                {
                    "microbatch": microbatch,
                    "lr": session.optimizer.learning_rate.item(),
                    "mlx": mx.random.uniform(shape=(4,)).tolist(),
                    "numpy": np.random.random(4).tolist(),
                    "python": [random.random() for _ in range(4)],
                }
            )
        original_save(session, microbatch)
        if args.stop and microbatch // session.accum == args.stop:
            raise KeyboardInterrupt("deliberate committed-update interruption")

    MLXSFTTrainerWrapper._apply_lora = apply
    MLXSFTTrainerWrapper.train = train
    FullStateSession.iterate = iterate
    FullStateSession.prepare = prepare
    FullStateSession.save = save

    class Tracker:
        def log_metrics(self, **values):
            trace.append(values)

    class Display:
        def start(self, total):
            pass

        def stop(self):
            pass

        def update(self, **values):
            pass

    try:
        if args.cli:
            from typer.testing import CliRunner

            from soup_cli.cli import app

            command = ["train", "--config", args.config, "--yes"]
            if args.resume:
                command += ["--resume", args.resume]
            outcome = CliRunner().invoke(app, command)
            # CliRunner turns KeyboardInterrupt into an explicit aborted invocation.
            if outcome.exit_code and not (args.stop or args.stop_micro):
                raise AssertionError((outcome.output, repr(outcome.exception)))
            result = captured.get("result", {"deliberate_interruption": True})
            captured["console"] = outcome.output
        else:
            wrapper = MLXSFTTrainerWrapper(cfg)
            wrapper.setup(json.loads(Path(args.rows).read_text()))
            result = wrapper.train(
                display=Display(),
                tracker=Tracker(),
                run_id="continuation-test",
                resume_from_checkpoint=args.resume,
            )
    except KeyboardInterrupt:
        if not (args.stop or args.stop_micro):
            raise
        result = {"deliberate_interruption": True}
    actual = dict(tree_flatten(captured["model"].parameters()))
    if "prepared" in captured:
        changed = any(
            not mx.array_equal(v, actual[k]).item() for k, v in captured["prepared"].items()
        )
        nonzero_rate = any(update["lr"] != 0 for update in updates)
        assert changed == nonzero_rate, "adapters must change after nonzero-rate updates"
    assert frozen and all(mx.array_equal(v, actual[k]).item() for k, v in frozen.items())
    Path(args.result).write_text(
        json.dumps(
            {
                "result": result,
                "trace": trace,
                "batches": batches,
                "updates": updates,
                "frozen_unchanged": True,
                "console": captured.get("console", ""),
            }
        )
    )


if __name__ == "__main__":
    main()
