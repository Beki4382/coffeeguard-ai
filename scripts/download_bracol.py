from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coffeeguard.config import BRACOL_ZIP_URL, RAW_DIR, ensure_dirs


def download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, out_path.open("wb") as f:
        total = response.headers.get("Content-Length")
        total_int = int(total) if total else None
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total_int:
                pct = downloaded / total_int * 100
                print(f"\rDownloaded {downloaded / 1024 / 1024:.1f} MB ({pct:.1f}%)", end="")
        print()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ensure_dirs()
    out_path = RAW_DIR / "bracol_mendeley.zip"
    if out_path.exists() and out_path.stat().st_size > 1_000_000:
        print(f"Using existing archive: {out_path}")
    else:
        print(f"Downloading BRACOL from {BRACOL_ZIP_URL}")
        download(BRACOL_ZIP_URL, out_path)
    print(f"Archive: {out_path}")
    print(f"Size MB: {out_path.stat().st_size / 1024 / 1024:.1f}")
    print(f"SHA256: {sha256(out_path)}")


if __name__ == "__main__":
    main()
