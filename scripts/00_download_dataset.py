import os
import shutil
import subprocess
import sys
from pathlib import Path

from asl import paths
from asl.data import find_train_dir

if paths.DATA.exists() and find_train_dir(paths.DATA):
    print("Dataset already present, skipping.")
    sys.exit(0)

has_token = (
    os.environ.get("KAGGLE_API_TOKEN")
    or (Path.home() / ".kaggle" / "access_token").exists()
    or (Path.home() / ".kaggle" / "kaggle.json").exists()
)
if not has_token:
    sys.exit(
        "Kaggle credentials not found.\n"
        "Create an API token at kaggle.com > Settings > API, then either save it to\n"
        "  ~/.kaggle/access_token   (token string) or ~/.kaggle/kaggle.json,\n"
        "or set the KAGGLE_API_TOKEN environment variable. Then rerun."
    )

kaggle = shutil.which("kaggle")
if kaggle is None:
    sys.exit("The 'kaggle' command was not found. Run: pip install -r requirements.txt")

paths.DATA.mkdir(exist_ok=True)
subprocess.run(
    [kaggle, "datasets", "download", "-d", "grassknoted/asl-alphabet",
     "-p", str(paths.DATA), "--unzip"],
    check=True,
)