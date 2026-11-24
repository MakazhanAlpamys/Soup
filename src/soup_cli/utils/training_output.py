"""Output directory construction shared by trainers and can validation."""

from pathlib import Path


def training_output_dir(
    output: str, experiment_name: str | None = None, *, backend: str = "", task: str = "",
) -> str:
    """Return the destination for the trainer's backend and task.

    The MLX trainers and unlearning write directly to the configured output.
    Other trainers use the experiment directory when one is configured.
    """
    if experiment_name and backend != "mlx" and task != "unlearn":
        return str(Path(output) / experiment_name)
    return output
