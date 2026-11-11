"""Real tiny GRPO runs with zero-spread and varied synthetic rewards.

Select one CUDA GPU externally. This is a functional monitor smoke, not a
model-quality or throughput benchmark. Output retains every measured log.
"""


def main() -> None:
    import argparse
    import json
    import logging
    from pathlib import Path

    import torch
    from datasets import Dataset
    from rich.console import Console
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast, set_seed
    from trl import GRPOConfig, GRPOTrainer

    from soup_cli.config.schema import SoupConfig
    from soup_cli.monitoring.callback import SoupTrainerCallback
    from soup_cli.monitoring.display import TrainingDisplay

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".codex/grpo-reward-monitor-smoke"))
    root = parser.parse_args().output
    root.mkdir(parents=True, exist_ok=True)
    rows = []

    class Capture(logging.Handler):
        def emit(self, record):
            rows.append(record.getMessage())

    logger = logging.getLogger("soup_cli.monitoring.callback")
    logger.addHandler(Capture())
    results = []
    for collapsed in (True, False):
        set_seed(2674)
        tokenizer = Tokenizer(
            WordLevel(
                {
                    "[PAD]": 0,
                    "[EOS]": 1,
                    "[UNK]": 2,
                    "hello": 3,
                    "world": 4,
                    "yes": 5,
                    "no": 6,
                    "test": 7,
                },
                unk_token="[UNK]",
            )
        )
        tokenizer.pre_tokenizer = Whitespace()
        tok = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]"
        )
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=8,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                pad_token_id=0,
                eos_token_id=1,
                bos_token_id=None,
                max_position_embeddings=64,
            )
        )
        config = SoupConfig(base="synthetic-tiny-llama", task="grpo", data={"train": "synthetic"})
        display = TrainingDisplay(config)
        display.start = lambda total_steps: setattr(display, "total_steps", total_steps)
        callback = SoupTrainerCallback(display=display)

        def rewards(completions, **kwargs):
            return [0.0 if collapsed else float(i % 2) for i in range(len(completions))]

        args = GRPOConfig(
            output_dir=str(root / str(collapsed)),
            max_steps=6,
            logging_steps=1,
            per_device_train_batch_size=2,
            num_generations=2,
            max_completion_length=4,
            beta=0.0,
            learning_rate=1e-5,
            bf16=True,
            report_to=[],
            save_strategy="no",
            gradient_checkpointing=False,
            disable_tqdm=True,
            seed=2674,
        )
        trainer = GRPOTrainer(
            model=model,
            args=args,
            reward_funcs=rewards,
            processing_class=tok,
            train_dataset=Dataset.from_dict({"prompt": ["hello world"] * 4}),
            callbacks=[callback],
        )
        before = len(rows)
        trainer.train()
        logs = [x for x in trainer.state.log_history if "frac_reward_zero_std" in x]
        warnings = [s for s in rows[before:] if "GRPO reward diversity is low" in s]
        assert len(logs) == 6, trainer.state.log_history
        assert len(warnings) == int(collapsed), warnings
        assert display.frac_reward_zero_std == float(collapsed)
        results.append(
            {
                "collapsed": collapsed,
                "steps": trainer.state.global_step,
                "metric_logs": logs,
                "warning_count": len(warnings),
                "display_fraction": display.frac_reward_zero_std,
            }
        )
        del trainer, model
        torch.cuda.empty_cache()
    result = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "trl": __import__("trl").__version__,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cases": results,
    }
    (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    Console().print(
        json.dumps(
            {
                "cases": [{k: v for k, v in x.items() if k != "metric_logs"} for x in results],
                "gpu": result["gpu"],
                "peak_allocated_bytes": result["peak_allocated_bytes"],
            }
        )
    )


if __name__ == "__main__":
    main()
