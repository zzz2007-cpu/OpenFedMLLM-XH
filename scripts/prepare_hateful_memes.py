#!/usr/bin/env python
"""Convert HF Hateful Memes data to image files plus JSONL annotations."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from io import BytesIO
from pathlib import Path
from typing import Any

from datasets import load_from_disk
from PIL import Image
from tqdm import tqdm


IMAGE_FIELDS = ("img", "image")
TEXT_FIELDS = ("text", "sentence")
LABEL_FIELDS = ("label", "labels")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare neuralcatcher/hateful_memes HF arrow data for multimodal training."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory saved by datasets.save_to_disk, e.g. ./hateful_memes_hf.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory, e.g. ./processed_hateful_memes.",
    )
    parser.add_argument(
        "--image_root",
        type=Path,
        default=None,
        help=(
            "Optional root directory for string image paths such as img/42953.png. "
            "If omitted, the script tries input_dir, input_dir parent, and cwd."
        ),
    )
    parser.add_argument(
        "--image_format",
        choices=("png", "jpg"),
        default="png",
        help="Image export format. Default: png.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output_dir first if it already exists; existing images are otherwise reused.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional per-split sample limit for debugging.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run a simple random sanity check after conversion.",
    )
    return parser.parse_args()


def find_field(columns: list[str], candidates: tuple[str, ...], required: bool, name: str) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    if required:
        raise ValueError(
            f"Missing required {name} field. Tried {list(candidates)}; available columns: {columns}"
        )
    return None


def safe_filename_stem(value: str) -> str:
    safe = value.replace("/", "_").replace("\\", "_").strip()
    return safe or "empty_id"


def get_sample_id(sample: dict[str, Any], split: str, index: int) -> str:
    sample_id = sample.get("id")
    if sample_id is None or str(sample_id).strip() == "":
        return f"{split}_{index:06d}"
    return str(sample_id)


def coerce_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def coerce_label(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def resolve_image_path(value: str | Path, search_dirs: list[Path]) -> Path:
    path = Path(value)
    candidates = [path]
    if not path.is_absolute():
        for root in search_dirs:
            candidates.append(root / path)
            if path.parts:
                candidates.append(root / path.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Image path {value!r} was not found. Checked: {checked}. "
        "If your dataset stores relative paths, pass --image_root pointing to the "
        "directory that contains the raw img/ folder."
    )


def coerce_image(value: Any, search_dirs: list[Path]) -> Image.Image:
    if isinstance(value, Image.Image):
        return value

    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(BytesIO(value["bytes"]))
        if value.get("path") is not None:
            return Image.open(resolve_image_path(value["path"], search_dirs))

    if isinstance(value, (str, Path)):
        return Image.open(resolve_image_path(value, search_dirs))

    if hasattr(value, "__array__"):
        return Image.fromarray(value)

    raise TypeError(f"Unsupported image field type: {type(value)!r}")


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "annotations").mkdir(parents=True, exist_ok=True)


def write_split(
    split: str,
    split_ds: Any,
    output_dir: Path,
    search_dirs: list[Path],
    image_format: str,
    overwrite: bool,
    limit: int | None,
) -> int:
    columns = list(split_ds.column_names)
    print(f"[{split}] columns: {columns}")

    image_field = find_field(columns, IMAGE_FIELDS, required=True, name="image")
    text_field = find_field(columns, TEXT_FIELDS, required=True, name="text")
    label_field = find_field(columns, LABEL_FIELDS, required=False, name="label")

    image_dir = output_dir / "images" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "annotations" / f"{split}.jsonl"

    split_len = len(split_ds)
    num_samples = min(split_len, limit) if limit is not None else split_len
    pil_format = "JPEG" if image_format == "jpg" else "PNG"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for index in tqdm(range(num_samples), desc=f"Processing {split}", unit="sample"):
            sample = split_ds[index]
            sample_id = get_sample_id(sample, split, index)
            image_name = f"{safe_filename_stem(sample_id)}.{image_format}"
            image_path = image_dir / image_name
            rel_image_path = image_path.relative_to(output_dir).as_posix()

            if overwrite or not image_path.exists():
                image = coerce_image(sample[image_field], search_dirs).convert("RGB")
                image.save(image_path, format=pil_format)

            text = coerce_text(sample[text_field])
            label = coerce_label(sample[label_field]) if label_field is not None else None
            record = {
                "id": sample_id,
                "image": rel_image_path,
                "text": text,
                "label": label,
                "split": split,
                "has_image": True,
                "has_text": bool(text.strip()),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"[{split}] samples={num_samples}, jsonl={jsonl_path}, images={image_dir}"
    )
    return num_samples


def write_dataset_info(output_dir: Path, splits: dict[str, int], image_format: str) -> None:
    info = {
        "dataset_name": "hateful_memes",
        "source": "neuralcatcher/hateful_memes",
        "splits": splits,
        "fields": {
            "id": "String sample id from the source data, or split_index fallback.",
            "image": "Image path relative to output_dir.",
            "text": "Meme text as a string.",
            "label": "Integer class label, or null when unavailable.",
            "split": "Dataset split name.",
            "has_image": "Whether the sample has an exported image.",
            "has_text": "Whether text is non-empty after stripping whitespace.",
        },
        "image_format": image_format,
        "total_samples": sum(splits.values()),
    }
    info_path = output_dir / "dataset_info.json"
    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sanity_check(output_dir: Path, splits: list[str]) -> None:
    print("[check] running random sanity checks")
    for split in splits:
        jsonl_path = output_dir / "annotations" / f"{split}.jsonl"
        records = read_jsonl(jsonl_path)
        if not records:
            print(f"[check][{split}] skipped: no records")
            continue

        record = random.choice(records)
        image_path = output_dir / record["image"]
        image_exists = image_path.exists()
        text_ok = isinstance(record.get("text"), str)
        label = record.get("label")
        label_ok = label is None or isinstance(label, int)

        if not image_exists:
            raise FileNotFoundError(f"[check][{split}] missing image: {image_path}")
        if not text_ok:
            raise TypeError(f"[check][{split}] text is not a string: {type(record.get('text'))!r}")
        if not label_ok:
            raise TypeError(f"[check][{split}] label is not int or None: {label!r}")

        print(
            f"[check][{split}] ok: id={record.get('id')}, "
            f"image={record.get('image')}, label={label}"
        )


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be non-negative.")

    prepare_output_dir(args.output_dir, args.overwrite)
    search_dirs = []
    if args.image_root is not None:
        search_dirs.append(args.image_root)
    search_dirs.extend([args.input_dir, args.input_dir.parent, Path.cwd()])
    search_dirs = [path.resolve() for path in search_dirs]

    ds = load_from_disk(str(args.input_dir))
    if not hasattr(ds, "keys"):
        ds = {"train": ds}

    splits: dict[str, int] = {}
    for split in ds.keys():
        splits[split] = write_split(
            split=split,
            split_ds=ds[split],
            output_dir=args.output_dir,
            search_dirs=search_dirs,
            image_format=args.image_format,
            overwrite=args.overwrite,
            limit=args.limit,
        )

    write_dataset_info(args.output_dir, splits, args.image_format)
    print(f"Total samples: {sum(splits.values())}")

    if args.check:
        sanity_check(args.output_dir, list(splits.keys()))


if __name__ == "__main__":
    main()
