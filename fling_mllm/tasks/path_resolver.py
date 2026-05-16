import os
from typing import List


def resolve_client_data_path(data_root: str, client_idx: int) -> str:
    """
    Resolve one client shard path with backward-compatible search order.

    Supported layouts:
    1) {data_root}/client_{i}.json
    2) {data_root}/client_{i}.jsonl
    3) {data_root}/client_{i}/   (directory shard, e.g. train_questions/train_annotations)
    """
    candidates: List[str] = [
        os.path.join(data_root, f"client_{client_idx}.json"),
        os.path.join(data_root, f"client_{client_idx}.jsonl"),
        os.path.join(data_root, f"client_{client_idx}"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    cand_text = "\n".join(candidates)
    raise FileNotFoundError(
        f"Client shard for client_{client_idx} not found under data_root={data_root!r}. "
        f"Tried:\n{cand_text}"
    )
