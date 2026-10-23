"""CPU reference subprocess; deliberately absent from the native runtime deps."""


def main():
    import hashlib
    import importlib.metadata
    import inspect
    import json
    import sys

    import peft.optimizers.lorafa as reference
    import torch

    assert importlib.metadata.version("peft") == "0.21.2"
    # Pin FP32 CPU behavior, avoiding device-dependent BF16 autocast.
    reference.is_bf16_available = lambda: False
    cases = json.load(sys.stdin)
    results = []
    for case in cases:
        a = torch.nn.Parameter(torch.tensor(case["a"]).T.contiguous(), requires_grad=False)
        b = torch.nn.Parameter(torch.tensor(case["b"]).T.contiguous())
        optimizer = reference.LoraFAOptimizer(
            [
                {
                    "params": [a, b],
                    "names": ["layer.lora_A.weight", "layer.lora_B.weight"],
                    "scaling_factors": [case["scale"], case["scale"]],
                }
            ],
            lr=0.003,
            weight_decay=0.07,
        )
        steps = []
        for gradient in case["gradients"]:
            b.grad = torch.tensor(gradient).T.contiguous()
            optimizer.step()
            state = optimizer.state["layer.lora"]
            steps.append(
                {
                    "b": b.T.tolist(),
                    "m": state["exp_avg_B"].T.tolist(),
                    "v": state["exp_avg_sq_B"].T.tolist(),
                }
            )
        results.append(steps)
    json.dump(
        {
            "cases": results,
            "version": importlib.metadata.version("peft"),
            "source_sha256": hashlib.sha256(inspect.getsource(reference).encode()).hexdigest(),
        },
        sys.stdout,
    )


if __name__ == "__main__":
    main()
