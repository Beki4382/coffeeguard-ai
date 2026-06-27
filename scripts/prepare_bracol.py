from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coffeeguard.config import CLASSES, INTERIM_DIR, PROCESSED_DIR, RAW_DIR, REPORTS_DIR, ensure_dirs
from coffeeguard.dataset import collect_bracol_leaf_csv, collect_labeled_images, stratified_split, write_audit, write_manifest


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir)
    except zipfile.BadZipFile:
        seven_zip = shutil.which("7z") or shutil.which("7zz")
        if not seven_zip:
            raise
        subprocess.run([seven_zip, "x", "-y", str(zip_path), f"-o{out_dir}"], check=False)


def extract_nested_archives(root: Path) -> None:
    for zip_path in sorted(root.rglob("*.zip")):
        target = zip_path.with_suffix("")
        if target.exists() and any(target.iterdir()):
            continue
        print(f"Extracting nested archive: {zip_path.name}")
        extract_zip(zip_path, target)


def copy_clean_images(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    clean_root = PROCESSED_DIR / "bracol_leaf"
    if clean_root.exists():
        shutil.rmtree(clean_root)
    clean_rows = []
    counters = {label: 0 for label in CLASSES}
    for row in rows:
        label = row["label"]
        counters[label] += 1
        src = Path(row["image_path"])
        dst = clean_root / label / f"{label}_{counters[label]:04d}{src.suffix.lower()}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        clean_rows.append({**row, "image_path": str(dst)})
    return clean_rows


def find_leaf_dirs(root: Path) -> list[Path]:
    candidates = []
    for csv_path in root.rglob("dataset.csv"):
        if csv_path.parent.name == "leaf" and (csv_path.parent / "images").exists():
            candidates.append(csv_path.parent)
    return candidates


def main() -> None:
    ensure_dirs()
    archive = RAW_DIR / "bracol_mendeley.zip"
    if not archive.exists():
        raise FileNotFoundError("Run scripts/download_bracol.py first.")

    extracted = INTERIM_DIR / "bracol_extracted"
    extract_zip(archive, extracted)
    extract_nested_archives(extracted)

    rows = []
    for leaf_dir in find_leaf_dirs(extracted):
        rows.extend(collect_bracol_leaf_csv(leaf_dir))
    if not rows:
        rows = collect_labeled_images(extracted)
    if not rows:
        raise RuntimeError(
            "No labeled images were discovered. Inspect data/interim/bracol_extracted and update label rules."
        )

    clean_rows = copy_clean_images(rows)
    split_rows = stratified_split(clean_rows)
    write_manifest(split_rows)
    write_audit(split_rows)

    print(f"Prepared {len(split_rows)} images")
    print(f"Manifest: {PROCESSED_DIR / 'bracol_splits.csv'}")
    print(f"Audit: {REPORTS_DIR / 'dataset_audit.csv'}")


if __name__ == "__main__":
    main()
