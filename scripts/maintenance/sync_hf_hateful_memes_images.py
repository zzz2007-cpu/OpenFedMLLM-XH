from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
import shutil

REPO_ID = "neuralcatcher/hateful_memes"
REPO_TYPE = "dataset"

LOCAL_IMG_DIR = Path("hateful_memes/img")
CACHE_DIR = Path("hateful_memes_hf_cache")

LOCAL_IMG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

api = HfApi()

print("Listing remote files...")
remote_files = api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)

remote_images = [
    f for f in remote_files
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
]

print(f"Remote image files: {len(remote_images)}")

local_names = {p.name.lower() for p in LOCAL_IMG_DIR.iterdir() if p.is_file()}

missing_remote_images = [
    f for f in remote_images
    if Path(f).name.lower() not in local_names
]

print(f"Missing local images to download: {len(missing_remote_images)}")

downloaded = 0
skipped = 0
failed = []

for remote_path in missing_remote_images:
    filename = Path(remote_path).name
    target_path = LOCAL_IMG_DIR / filename

    if target_path.exists():
        skipped += 1
        continue

    try:
        print(f"Downloading {remote_path} -> {target_path}")

        cached_file = hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=remote_path,
            cache_dir=str(CACHE_DIR),
        )

        shutil.copy2(cached_file, target_path)
        downloaded += 1

    except Exception as e:
        print(f"FAILED: {remote_path}: {e}")
        failed.append((remote_path, str(e)))

print("\nDone.")
print(f"Downloaded: {downloaded}")
print(f"Skipped: {skipped}")
print(f"Failed: {len(failed)}")

if failed:
    report = Path("outputs/analysis/hf_image_download_failed.txt")
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", encoding="utf-8") as f:
        for path, err in failed:
            f.write(f"{path}\t{err}\n")
    print(f"Failure report saved to: {report}")