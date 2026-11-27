"""Bounded local text preprocessing; all spool state is private and disposable."""

from __future__ import annotations

import json
import math
import random
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from soup_cli.utils.paths import is_under_cwd
from soup_cli.utils.terminal import strip_control

_TEXT_FORMATS = {"chatml", "alpaca", "sharegpt", "plaintext", "input_output", "tool-calling"}


class _DiskOrder:
    """Mutable row references for the existing RNG's Fisher-Yates shuffle."""

    def __init__(self, connection: sqlite3.Connection, length: int) -> None:
        self.connection = connection
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[int, int]:
        return self.connection.execute(
            "SELECT source, row_position FROM ordering WHERE position = ?", (index,),
        ).fetchone()

    def __setitem__(self, index: int, value: tuple[int, int]) -> None:
        self.connection.execute(
            "UPDATE ordering SET source = ?, row_position = ? WHERE position = ?",
            (*value, index),
        )


def _mix_replay(connection: sqlite3.Connection, count: int, pool_size: int, config) -> None:
    from soup_cli.data.loader import console
    from soup_cli.utils.rehearsal import resolve_replay_count

    requested = resolve_replay_count(count, config.replay_ratio)
    take = min(requested, pool_size)
    rng = random.Random(config.replay_seed if config.replay_seed is not None else 0)
    # Match Random.sample's pool/set choice and draw order before the final shuffle.
    set_size = 21 + (4 ** math.ceil(math.log(take * 3, 4)) if take > 5 else 0)
    if pool_size <= set_size:
        connection.execute("CREATE TABLE pool (position INTEGER PRIMARY KEY, value INTEGER)")
        connection.execute("INSERT INTO pool SELECT position, position FROM rows WHERE source = 1")
    else:
        connection.execute("CREATE TABLE selected (position INTEGER PRIMARY KEY)")
    for index in range(take):
        if pool_size <= set_size:
            selected = rng.randrange(pool_size - index)
            chosen = connection.execute(
                "SELECT value FROM pool WHERE position = ?", (selected,),
            ).fetchone()[0]
            connection.execute(
                "UPDATE pool SET value = (SELECT value FROM pool WHERE position = ?) "
                "WHERE position = ?", (pool_size - index - 1, selected),
            )
        else:
            chosen = rng.randrange(pool_size)
            while not connection.execute(
                "INSERT OR IGNORE INTO selected VALUES (?)", (chosen,),
            ).rowcount:
                chosen = rng.randrange(pool_size)
        connection.execute("INSERT INTO ordering VALUES (?, 1, ?)", (count + index, chosen))
    if take:
        order = _DiskOrder(connection, count + take)
        for index in reversed(range(1, len(order))):
            selected = rng.randrange(index + 1)
            order[index], order[selected] = order[selected], order[index]
    final_count = count + take
    console.print(
        f"[dim]Replay: +{take} old rows interleaved "
        f"({(take / final_count if final_count else 0.0) * 100:.1f}% of {final_count})[/]"
    )
    if requested > take:
        console.print(
            f"[yellow]Replay pool too small: wanted {requested}, used {take} "
            f"(short {requested - take}). Rows are NOT repeated.[/]"
        )


def _read_rows(connection: sqlite3.Connection, source: int) -> Iterator[dict]:
    with closing(connection.execute(
        "SELECT data FROM raw WHERE source = ? ORDER BY position", (source,),
    )) as cursor:
        for (data,) in cursor:
            yield json.loads(data)


@contextmanager
def preprocessing_rows(config, task: str):
    """Keep unsupported sources on the existing loader, including its media checks."""
    from soup_cli.data import loader
    from soup_cli.data.formats import MAX_DETECT_ROWS, detect_format

    paths = [config.train] + ([config.replay] if config.replay else [])
    eligible = (
        task in {"sft", "pretrain"}
        and config.format in _TEXT_FORMATS | {"auto"}
        and all(isinstance(path, str) and "://" not in path
                and Path(path).suffix.lower() == ".jsonl" for path in paths)
    )
    if not eligible:
        yield loader.load_dataset(config).get("train", [])
        return

    with TemporaryDirectory(prefix="soup-preprocess-rows-") as folder:
        with closing(sqlite3.connect(str(Path(folder) / "rows.sqlite"))) as connection:
            connection.execute("PRAGMA cache_size = -1024")
            connection.execute("PRAGMA temp_store = FILE")
            connection.execute(
                "CREATE TABLE raw (source INTEGER, position INTEGER, data TEXT, "
                "PRIMARY KEY (source, position)) WITHOUT ROWID"
            )
            connection.execute(
                "CREATE TABLE rows (source INTEGER, position INTEGER, data TEXT, "
                "PRIMARY KEY (source, position)) WITHOUT ROWID"
            )
            loader._last_load = loader.LoadOutcome()
            supported = True
            try:
                with loader.console.capture() as captured:
                    counts = []
                    for source, name in enumerate(paths):
                        path = Path(name)
                        if source and not is_under_cwd(path):
                            raise ValueError(
                                f"data.replay path is outside the working directory: {path}"
                            )
                        if not path.exists():
                            prefix = "data.replay file" if source else "Data file"
                            raise FileNotFoundError(f"{prefix} not found: {path}")
                        sample: list[dict] = []
                        for position, row in enumerate(loader._iter_jsonl(path)):
                            if len(sample) < MAX_DETECT_ROWS:
                                sample.append(row)
                            connection.execute(
                                "INSERT INTO raw VALUES (?, ?, ?)",
                                (source, position, json.dumps(row)),
                            )
                        fmt = config.format if source == 0 else "auto"
                        if fmt == "auto":
                            fmt = detect_format(sample)
                        if fmt not in _TEXT_FORMATS:
                            supported = False
                            break
                        if source == 0 and config.format == "auto":
                            loader.console.print(f"[dim]Auto-detected format: {fmt}[/]")
                        count = 0
                        for count, row in enumerate(loader._iter_formatted_rows(
                            _read_rows(connection, source), fmt, source=str(path),
                        ), start=1):
                            connection.execute(
                                "INSERT INTO rows VALUES (?, ?, ?)",
                                (source, count - 1, json.dumps(row)),
                            )
                        counts.append(count)
                        connection.execute("DELETE FROM raw WHERE source = ?", (source,))
            except BaseException:
                loader.console.print(
                    strip_control(captured.get()), end="", markup=False, highlight=False,
                )
                raise
            if not supported:
                yield loader.load_dataset(config).get("train", [])
                return
            loader.console.print(
                strip_control(captured.get()), end="", markup=False, highlight=False,
            )
            train_count = int(counts[0] * (1 - config.val_split))
            connection.execute(
                "CREATE TABLE ordering (position INTEGER PRIMARY KEY, "
                "source INTEGER, row_position INTEGER)"
            )
            connection.execute(
                "INSERT INTO ordering SELECT position, 0, position FROM rows "
                "WHERE source = 0 AND position < ?", (train_count,),
            )
            if config.replay:
                _mix_replay(connection, train_count, counts[1], config)
            connection.commit()
            with closing(connection.execute(
                "SELECT rows.data FROM ordering JOIN rows "
                "ON rows.source = ordering.source AND rows.position = ordering.row_position "
                "ORDER BY ordering.position"
            )) as cursor:
                rows = (json.loads(data) for (data,) in cursor)
                try:
                    yield rows
                finally:
                    rows.close()
