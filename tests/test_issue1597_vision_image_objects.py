"""Tests for Issue #1597: Vision SFT on Hub/Parquet datasets with decoded Image or struct."""

import io
import tempfile
from pathlib import Path

from PIL import Image

from soup_cli.config.schema import DataConfig
from soup_cli.data.loader import _validate_vision_images, load_dataset
from soup_cli.trainer.sft import SFTTrainerWrapper


def test_prepare_vision_dataset_with_pil_images():
    """Hub datasets return in-memory PIL Images, which should be accepted and converted to RGB."""
    img1 = Image.new("RGBA", (16, 16), (255, 0, 0, 128))
    img2 = Image.new("RGB", (16, 16), "blue")
    dataset = {
        "train": [
            {"image": img1, "messages": [{"role": "user", "content": "hi"}]},
            {"image": img2, "messages": [{"role": "user", "content": "hello"}]},
        ],
        "val": [
            {"image": img1, "messages": [{"role": "user", "content": "val"}]},
        ],
    }

    wrapper = SFTTrainerWrapper.__new__(SFTTrainerWrapper)
    train_ds, val_ds = wrapper._prepare_vision_dataset(dataset)

    assert len(train_ds) == 2
    assert len(train_ds[0]["images"]) == 1
    assert train_ds[0]["images"][0].mode == "RGB"
    assert len(train_ds[1]["images"]) == 1
    assert train_ds[1]["images"][0].mode == "RGB"

    assert len(val_ds) == 1
    assert len(val_ds[0]["images"]) == 1
    assert val_ds[0]["images"][0].mode == "RGB"


def test_prepare_vision_dataset_with_image_bytes_struct():
    """Parquet datasets can store images as dicts with raw bytes."""
    buf = io.BytesIO()
    Image.new("RGB", (10, 10), "green").save(buf, format="PNG")
    png_bytes = buf.getvalue()

    dataset = {
        "train": [
            {
                "image": {"bytes": png_bytes, "path": "test.png"},
                "messages": [{"role": "user", "content": "hi"}],
            },
        ]
    }

    wrapper = SFTTrainerWrapper.__new__(SFTTrainerWrapper)
    train_ds, val_ds = wrapper._prepare_vision_dataset(dataset)

    assert len(train_ds) == 1
    assert len(train_ds[0]["images"]) == 1
    assert train_ds[0]["images"][0].size == (10, 10)
    assert train_ds[0]["images"][0].mode == "RGB"
    assert val_ds is None


def test_validate_vision_images_with_pil_and_structs():
    """Validate that _validate_vision_images accepts in-memory images and enforces traversal."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        local_img = base_dir / "valid.png"
        Image.new("RGB", (8, 8), "white").save(local_img)

        pil_img = Image.new("RGB", (8, 8), "red")
        data = [
            {"image": pil_img, "id": "1"},
            {"image": {"bytes": b"fakebytes", "path": None}, "id": "2"},
            {"image": {"bytes": None, "path": "valid.png"}, "id": "3"},
            {"image": {"bytes": None, "path": "../outside.png"}, "id": "4"},
            {"image": str(local_img), "id": "5"},
            {"image": "/etc/passwd", "id": "6"},
        ]

        validated = _validate_vision_images(data, base_dir)
        ids = [row["id"] for row in validated]
        # Rows 1 (PIL), 2 (bytes), 3 (valid struct path), 5 (valid path str) kept
        # Rows 4 (traversal struct) and 6 (traversal str) rejected
        assert ids == ["1", "2", "3", "5"]


def test_load_parquet_with_image_features():
    """Loading a parquet dataset containing datasets.Image column should work end-to-end."""
    import datasets

    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = Path(tmpdir) / "train.parquet"
        img = Image.new("RGB", (8, 8), "red")
        turns = [{"from": "human", "value": "<image>\nColour?"}, {"from": "gpt", "value": "Red."}]

        ds = datasets.Dataset.from_dict(
            {"image": [img], "conversations": [turns]},
            features=datasets.Features({
                "image": datasets.Image(),
                "conversations": [
                    {"from": datasets.Value("string"), "value": datasets.Value("string")}
                ],
            }),
        )
        ds.to_parquet(str(parquet_path))

        loaded = load_dataset(DataConfig(train=str(parquet_path), format="llava", val_split=0.0))
        assert len(loaded["train"]) == 1
        assert "image" in loaded["train"][0]

        # And feeding to _prepare_vision_dataset should produce images
        wrapper = SFTTrainerWrapper.__new__(SFTTrainerWrapper)
        train_ds, _ = wrapper._prepare_vision_dataset(loaded)
        assert len(train_ds) == 1
        assert len(train_ds[0]["images"]) == 1


def test_validate_vision_images_resolves_struct_path(tmp_path):
    Image.new("RGB", (8, 8), "white").save(tmp_path / "valid.png")
    rows = [{"image": {"bytes": None, "path": "valid.png"}}]
    validated = _validate_vision_images(rows, tmp_path)
    assert validated[0]["image"]["path"] == str((tmp_path / "valid.png").resolve())


def test_prepare_vision_dataset_opens_path_only_struct(tmp_path):
    img = tmp_path / "pic.png"
    Image.new("RGB", (6, 4), "blue").save(img)
    dataset = {"train": [{"image": {"bytes": None, "path": str(img)},
                          "messages": [{"role": "user", "content": "hi"}]}]}
    wrapper = SFTTrainerWrapper.__new__(SFTTrainerWrapper)
    train_ds, _ = wrapper._prepare_vision_dataset(dataset)
    assert [im.size for im in train_ds[0]["images"]] == [(6, 4)]

