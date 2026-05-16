import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ..registry import register_data_loader


PROMPT_TEMPLATE = """Given the image and text, determine whether the meme is hateful.

Answer with only:
yes
or
no.

Text:
{text}"""

LABEL_VERBALIZER = {
    0: "no",
    1: "yes",
}


def _read_jsonl(path: Path) -> List[Dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} at line {line_no}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Expected object in {path} at line {line_no}, got {type(item).__name__}")
            records.append(item)
    return records


def _read_records(path: Path) -> List[Dict]:
    if path.suffix.lower() == ".jsonl":
        return _read_jsonl(path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Hateful Memes data must be a JSON list or JSONL file: {path}")


def _candidate_roots(data_path: Path, explicit_root: Optional[str]) -> Iterable[Path]:
    if explicit_root:
        yield Path(explicit_root).expanduser().resolve()
    for parent in [data_path.parent, *data_path.parents]:
        yield parent


def infer_hateful_memes_root(data_path: str, explicit_root: Optional[str] = None) -> Path:
    path = Path(data_path).expanduser().resolve()
    for candidate in _candidate_roots(path, explicit_root):
        if (candidate / "img").is_dir() and (
            (candidate / "train.jsonl").exists()
            or (candidate / "federated").is_dir()
        ):
            return candidate
    if explicit_root:
        root = Path(explicit_root).expanduser().resolve()
        raise FileNotFoundError(
            f"hateful_memes_root does not look valid: {root}. "
            "Expected an img/ directory and train.jsonl or federated/."
        )
    raise FileNotFoundError(
        f"Could not infer Hateful Memes root from data_path={data_path!r}. "
        "Set data_args.hateful_memes_root or OPENFED_HATEFUL_MEMES_ROOT."
    )


def _normalize_label(raw_label, sample_id: str) -> str:
    try:
        label = int(raw_label)
    except Exception as exc:
        raise ValueError(f"Hateful Memes sample {sample_id} has non-integer label: {raw_label!r}") from exc
    if label not in LABEL_VERBALIZER:
        raise ValueError(f"Hateful Memes sample {sample_id} has invalid label {label!r}; expected 0 or 1.")
    return LABEL_VERBALIZER[label]


def _resolve_image_path(raw_img, root: Path, sample_id: str, strict_image_path: bool):
    if raw_img is None or str(raw_img).strip() == "":
        return None
    img_path = Path(str(raw_img))
    if img_path.is_absolute():
        resolved = img_path
    else:
        resolved = root / img_path
    if strict_image_path and not resolved.exists():
        raise FileNotFoundError(
            f"Hateful Memes image not found for sample {sample_id}: "
            f"raw img={raw_img!r}, resolved path={resolved}"
        )
    if not resolved.exists():
        return None
    return str(resolved)


def format_hateful_memes_sample(
    sample: Dict,
    root: Path,
    require_answer: bool = True,
    strict_image_path: bool = True,
) -> Dict:
    sample_id = str(sample.get("id", "")).strip()
    if not sample_id:
        raise ValueError("Hateful Memes sample is missing required field 'id'.")

    text = sample.get("text")
    text = "" if text is None else str(text)
    question = PROMPT_TEMPLATE.format(text=text)
    image_path = _resolve_image_path(
        raw_img=sample.get("img", sample.get("image")),
        root=root,
        sample_id=sample_id,
        strict_image_path=strict_image_path,
    )

    out = {
        "id": sample_id,
        "image": image_path,
        "question": question,
        "text": text,
        "metadata": {
            "dataset": "hateful_memes",
            "raw_img": sample.get("img", sample.get("image")),
        },
    }

    if "label" in sample and sample.get("label") is not None:
        answer = _normalize_label(sample.get("label"), sample_id=sample_id)
        out["answer"] = answer
        out["label"] = int(sample.get("label"))
        out["conversations"] = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    elif require_answer:
        raise ValueError(f"Hateful Memes sample {sample_id} is missing required field 'label'.")
    else:
        out["conversations"] = [
            {"role": "user", "content": question},
        ]
    return out


@register_data_loader("hateful_memes")
def load_hateful_memes_samples(
    data_path: str,
    split: str = "train",
    data_format: str = "auto",
    hateful_memes_root: Optional[str] = None,
    require_answer: Optional[bool] = None,
    strict_image_path: bool = True,
    **kwargs,
) -> List[Dict]:
    path = Path(data_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Hateful Memes data path not found: {path}")
    if path.is_dir():
        raise ValueError(
            f"Hateful Memes loader expects a client_*.jsonl/json file or raw split file, got directory: {path}"
        )
    if require_answer is None:
        require_answer = str(split or "train").lower() in {"train", "training"}

    root = infer_hateful_memes_root(str(path), explicit_root=hateful_memes_root)
    raw_records = _read_records(path)
    return [
        format_hateful_memes_sample(
            sample=record,
            root=root,
            require_answer=bool(require_answer),
            strict_image_path=bool(strict_image_path),
        )
        for record in raw_records
    ]
