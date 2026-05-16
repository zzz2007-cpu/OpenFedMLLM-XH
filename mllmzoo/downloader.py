import hashlib
import os
from huggingface_hub import snapshot_download


def get_cache_dir():
    return os.path.join(os.path.expanduser("~"), ".cache", "mllmzoo")


def verify_checksum(file_path: str, expected_sha256: str):
    if not expected_sha256:
        return True
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest() == expected_sha256


def download_from_hf(repo_id: str, cache_subdir: str = "models"):
    local_dir = os.path.join(get_cache_dir(), cache_subdir, repo_id.replace("/", "__"))
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False, resume_download=True)
    return local_dir

