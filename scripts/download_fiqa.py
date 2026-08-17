"""Safely and idempotently acquire the canonical BEIR FiQA archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import ssl
import stat
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import certifi

from _phase1_common import PROJECT_ROOT, load_config, repository_path
from qkef.datasets.fiqa import FiqaDataError, load_fiqa_dataset, validate_fiqa_layout


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(url: str, archive_path: Path, expected_md5: str, *, force: bool) -> str:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        observed = md5_file(archive_path)
        if observed == expected_md5:
            print(f"Archive: reuse verified {archive_path.relative_to(PROJECT_ROOT)}")
            return observed
        if not force:
            raise RuntimeError(
                f"existing archive MD5 mismatch ({observed}); rerun with --force to replace it"
            )
        archive_path.unlink()
    partial = archive_path.with_suffix(archive_path.suffix + ".part")
    if partial.exists():
        partial.unlink()
    request = urllib.request.Request(url, headers={"User-Agent": "Q-KEF-Phase1/1.0"})
    try:
        tls_context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(request, timeout=60, context=tls_context) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise
    observed = md5_file(partial)
    if observed != expected_md5:
        partial.unlink()
        raise RuntimeError(
            f"downloaded archive MD5 mismatch: expected {expected_md5}, observed {observed}"
        )
    partial.replace(archive_path)
    print(f"Archive: downloaded and verified {archive_path.relative_to(PROJECT_ROOT)}")
    return observed


def _validate_member(member: zipfile.ZipInfo, destination: Path) -> None:
    name = PurePosixPath(member.filename)
    if name.is_absolute() or ".." in name.parts:
        raise RuntimeError(f"unsafe ZIP member path: {member.filename!r}")
    mode = member.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise RuntimeError(f"symbolic links are not allowed in dataset ZIP: {member.filename!r}")
    target = (destination / Path(*name.parts)).resolve()
    try:
        target.relative_to(destination.resolve())
    except ValueError as exc:
        raise RuntimeError(f"ZIP member escapes extraction root: {member.filename!r}") from exc


def safe_extract_dataset(archive_path: Path, dataset_dir: Path, *, force: bool) -> None:
    try:
        validate_fiqa_layout(dataset_dir)
        print(f"Dataset: reuse validated {dataset_dir.relative_to(PROJECT_ROOT)}")
        return
    except FiqaDataError:
        if dataset_dir.exists() and not force:
            raise RuntimeError(
                f"existing extraction is incomplete: {dataset_dir}; rerun with --force to replace it"
            )
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fiqa-extract-", dir=dataset_dir.parent) as temporary_name:
        temporary = Path(temporary_name)
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                _validate_member(member, temporary)
            archive.extractall(temporary)
        candidates = []
        for corpus_path in temporary.rglob("corpus.jsonl"):
            candidate = corpus_path.parent
            try:
                validate_fiqa_layout(candidate)
            except FiqaDataError:
                continue
            candidates.append(candidate)
        if len(candidates) != 1:
            raise RuntimeError(
                f"expected one FiQA dataset root in archive, found {len(candidates)}"
            )
        shutil.copytree(candidates[0], dataset_dir)
    validate_fiqa_layout(dataset_dir)
    print(f"Dataset: safely extracted {dataset_dir.relative_to(PROJECT_ROOT)}")


def write_source_metadata(
    dataset_dir: Path,
    archive_path: Path,
    source_url: str,
    expected_md5: str,
    observed_md5: str,
    metadata_path: Path,
    seed: int,
) -> dict[str, object]:
    dataset = load_fiqa_dataset(dataset_dir)
    split_rows = {split: len(rows) for split, rows in dataset.qrels_by_split.items()}
    split_queries = {
        split: len({row.query_id for row in rows})
        for split, rows in dataset.qrels_by_split.items()
    }
    existing_timestamp = None
    if metadata_path.exists():
        try:
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            if existing.get("observed_md5") == observed_md5:
                existing_timestamp = existing.get("acquisition_timestamp_utc")
        except (OSError, json.JSONDecodeError):
            pass
    metadata: dict[str, object] = {
        "dataset_name": "fiqa",
        "benchmark_name": "BEIR",
        "source_type": "official public benchmark archive",
        "source_url": source_url,
        "archive_filename": archive_path.name,
        "expected_md5": expected_md5,
        "observed_md5": observed_md5,
        "checksum_verified": observed_md5 == expected_md5,
        "acquisition_timestamp_utc": existing_timestamp
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "corpus_count": len(dataset.documents),
        "query_count": len(dataset.queries),
        "qrels_split_names": sorted(dataset.qrels_by_split),
        "qrel_row_counts": split_rows,
        "unique_query_counts_by_qrels_split": split_queries,
        "local_paths": {
            "archive": archive_path.relative_to(PROJECT_ROOT).as_posix(),
            "dataset_root": dataset_dir.relative_to(PROJECT_ROOT).as_posix(),
        },
        "project_seed": seed,
        "notes": [
            "Observed counts are recorded from the checksum-verified archive.",
            "Acquisition does not alter corpus, query, or qrels content.",
        ],
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(metadata_path)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--force", action="store_true", help="replace an invalid archive/extraction")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config, _ = load_config(args.config)
    dataset_config = config["dataset"]
    raw_dir = repository_path(dataset_config["raw_dir"])
    archive_path = raw_dir / "fiqa.zip"
    dataset_dir = raw_dir / "fiqa"
    expected_md5 = str(dataset_config["expected_md5"])
    observed_md5 = download_archive(
        str(dataset_config["source_url"]), archive_path, expected_md5, force=args.force
    )
    safe_extract_dataset(archive_path, dataset_dir, force=args.force)
    metadata = write_source_metadata(
        dataset_dir,
        archive_path,
        str(dataset_config["source_url"]),
        expected_md5,
        observed_md5,
        PROJECT_ROOT / "data" / "metadata" / "fiqa_source.json",
        int(config["project"]["seed"]),
    )
    print(
        f"FiQA acquisition: SUCCESS ({metadata['corpus_count']} documents, "
        f"{metadata['query_count']} queries)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
