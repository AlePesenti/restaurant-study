"""Move the pre-rendered audio between this machine and a Hugging Face dataset.

The audio (about 180 MB) is too large for the GitHub repository, so it lives in a public
Hugging Face dataset instead. Render downloads it during the build (see render.yaml).

    python webapp/audio_dataset.py upload   <user>/<dataset>   # after webapp/precompute.py (needs HF_TOKEN)
    python webapp/audio_dataset.py download <user>/<dataset>   # run by the Render build
"""

import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

CACHE_DIR = Path(__file__).resolve().parent / "audio_cache"


def upload(dataset_id):
    api = HfApi()
    api.create_repo(dataset_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path=CACHE_DIR, repo_id=dataset_id, repo_type="dataset",
                      allow_patterns="*/*.mp3", commit_message="Update audio cache")
    print(f"Uploaded to https://huggingface.co/datasets/{dataset_id}")


def download(dataset_id):
    snapshot_download(dataset_id, repo_type="dataset", local_dir=CACHE_DIR, allow_patterns="*/*.mp3")
    print(f"{len(list(CACHE_DIR.glob('*/*.mp3')))} audio files in {CACHE_DIR}")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("upload", "download"):
        sys.exit(__doc__)
    {"upload": upload, "download": download}[sys.argv[1]](sys.argv[2])
