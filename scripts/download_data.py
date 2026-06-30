"""Download Kaggle competition data to data/raw/ and unzip."""
import subprocess
import zipfile
from pathlib import Path

COMPETITION = "playground-series-s6e2"
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "kaggle", "competitions", "download",
            "-c", COMPETITION,
            "-p", str(DATA_DIR),
        ],
        check=True,
    )
    for zip_path in DATA_DIR.glob("*.zip"):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(DATA_DIR)
        zip_path.unlink()
    print(f"Data downloaded to {DATA_DIR}")
    print("Files:")
    for f in sorted(DATA_DIR.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
