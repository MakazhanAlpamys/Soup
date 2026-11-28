"""Shared fixture for the streamed-layer residency tests: a tiny six-layer streamed stack."""

from types import SimpleNamespace


def fixture(device="cpu"):
    import torch

    from soup_cli.utils.layer_stream_runtime import (
        LayerBufferPool,
        StreamedDecoderLayer,
        StreamPrefetcher,
    )

    class Source:
        nbytes = 6 * 4 * 4 * 4

        def __init__(self):
            self.calls = []
            self.weights = [torch.eye(4) * (0.5 + idx / 16) for idx in range(6)]
            if device == "cuda":
                self.weights = [value.pin_memory() for value in self.weights]

        def get(self, idx, name):
            assert name == "base"
            self.calls.append(idx)
            return self.weights[idx]

    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Parameter(torch.empty(4, 4, device="meta"), requires_grad=False)
            self.adapter = torch.nn.Parameter(torch.full((4, 4), 0.03125, device=device))

        def forward(self, hidden):
            return (hidden @ self.base.T + hidden @ self.adapter.T).tanh()

    source = Source()
    pool = LayerBufferPool({"base": ((4, 4), "float32")}, device=device)
    stream = torch.cuda.Stream() if device == "cuda" else None
    prefetcher = StreamPrefetcher(pool, source, 6, stream=stream)
    model = torch.nn.Sequential(*[
        StreamedDecoderLayer(Inner(), idx, pool, prefetcher, {"base": "base"})
        for idx in range(6)
    ])
    runtime = SimpleNamespace(pool=pool, source=source, prefetcher=prefetcher,
                              n_layers=6, tier="ram")
    return model, runtime


def trajectory(model, runtime):
    import torch

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
    outputs = []
    for _step in range(4):
        runtime.prefetcher.prime()
        device = runtime.pool.buffers[0]["base"].device
        hidden = (torch.arange(8, device=device).reshape(2, 4).float() / 16).requires_grad_(True)
        result = model(hidden)
        loss = result.square().mean()
        loss.backward()
        outputs.extend([result.detach().clone(), loss.detach().clone(), hidden.grad.clone()])
        outputs.extend(p.grad.clone() for p in model.parameters() if p.requires_grad)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    outputs.extend(p.detach().clone() for p in model.parameters() if p.requires_grad)
    return outputs
