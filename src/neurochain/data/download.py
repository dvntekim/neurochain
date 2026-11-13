"""Download, verify and unpack the registered datasets.

Usage::

    neurochain-data fetch              # every dataset that can be scripted
    neurochain-data fetch neuralink    # one dataset
    neurochain-data status             # what is on disk and whether it verifies

Design rules:

* A file is only moved into place after its SHA-256 matches the registry, so a
  half-finished or tampered download can never be mistaken for data.
* Datasets that need a browser are reported with instructions instead of
  failing the whole run, so ``make u0`` works with only the Neuralink data.
* Zip members are written by base name only, which rules out path traversal.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.error
import urllib.request
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from neurochain.data.registry import DATASETS, Dataset, RemoteFile, get_dataset
from neurochain.paths import raw_dir

USER_AGENT = "neurochain-data/0.1"
CHUNK_BYTES = 1 << 20
CLIPS_SUBDIR = "clips"

FileState = Literal["ok", "unverified", "missing", "corrupt"]


class DownloadError(RuntimeError):
    """A file could not be fetched from any of its URLs."""


class ChecksumError(DownloadError):
    """A fetched file does not match its registered SHA-256."""


@dataclass(frozen=True)
class FileStatus:
    """Verification result for one file on disk.

    Attributes:
        file: The registry entry.
        path: Where the file is expected.
        state: ``ok`` (hash matches), ``unverified`` (present, no registered
            hash), ``missing`` or ``corrupt`` (hash mismatch).
        sha256: Digest of the file on disk, when present.
    """

    file: RemoteFile
    path: Path
    state: FileState
    sha256: str | None = None


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 of a file, reading it in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def file_status(dataset: Dataset, file: RemoteFile) -> FileStatus:
    """Check one registered file against what is on disk."""
    path = raw_dir(dataset.key) / file.name
    if not path.is_file():
        return FileStatus(file, path, "missing")
    actual = sha256_of(path)
    if file.sha256 is None:
        return FileStatus(file, path, "unverified", actual)
    state: FileState = "ok" if actual == file.sha256 else "corrupt"
    return FileStatus(file, path, state, actual)


def fetch_url(
    url: str,
    dest: Path,
    *,
    expected_sha256: str | None = None,
    token: str | None = None,
    timeout_s: float = 60.0,
    show_progress: bool = True,
) -> str:
    """Stream ``url`` to ``dest`` and return the SHA-256 of what was written.

    The bytes go to ``dest.part`` first and are renamed only once the download
    is complete and, if ``expected_sha256`` is given, the digest matches.

    Args:
        url: Source URL.
        dest: Final path of the file.
        expected_sha256: Hex digest the download must match.
        token: Bearer token to send, for hosts that require one.
        timeout_s: Socket timeout in seconds.
        show_progress: Print coarse progress to standard error.

    Raises:
        ChecksumError: If the digest does not match ``expected_sha256``.
        urllib.error.URLError: On network or HTTP failure.
    """
    headers = {"User-Agent": USER_AGENT}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response, part.open("wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            next_report = 0.1
            while chunk := response.read(CHUNK_BYTES):
                out.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if show_progress and total and done / total >= next_report:
                    print(
                        f"    {done / 1e6:7.1f} / {total / 1e6:.1f} MB",
                        file=sys.stderr,
                        flush=True,
                    )
                    next_report += 0.1
        actual = digest.hexdigest()
        if expected_sha256 is not None and actual != expected_sha256:
            raise ChecksumError(
                f"{url}: SHA-256 {actual} does not match the registered {expected_sha256}"
            )
        os.replace(part, dest)
    finally:
        part.unlink(missing_ok=True)
    return actual


def fetch_file(dataset: Dataset, file: RemoteFile, *, show_progress: bool = True) -> FileStatus:
    """Make sure one file is on disk and verified, downloading it if needed.

    Raises:
        DownloadError: If the file is absent and cannot be fetched.
    """
    status = file_status(dataset, file)
    if status.state in ("ok", "unverified"):
        return status
    if status.state == "corrupt":
        raise ChecksumError(
            f"{status.path} exists but its SHA-256 is {status.sha256}, "
            f"expected {file.sha256}. Delete it and fetch again."
        )
    token = os.environ.get(dataset.token_env) if dataset.token_env else None
    if not file.urls or (dataset.token_env and not token):
        raise DownloadError(f"{file.name} needs a manual download")
    errors: list[str] = []
    for url in file.urls:
        print(f"  fetching {file.name} from {url}", file=sys.stderr)
        try:
            fetch_url(
                url,
                status.path,
                expected_sha256=file.sha256,
                token=token,
                show_progress=show_progress,
            )
        except (urllib.error.URLError, TimeoutError, ChecksumError) as exc:
            errors.append(f"{url}: {exc}")
            print(f"    failed: {exc}", file=sys.stderr)
            continue
        return file_status(dataset, file)
    raise DownloadError(f"could not fetch {file.name}:\n  " + "\n  ".join(errors))


def safe_extract_zip(zip_path: Path, dest_dir: Path, *, suffix: str = ".wav") -> int:
    """Extract files with the given suffix from a zip, flattened into ``dest_dir``.

    Each member is written under its base name, so an archive cannot place
    files outside ``dest_dir`` whatever paths it declares.

    Returns:
        The number of files extracted.

    Raises:
        ValueError: If two members share a base name.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            name = Path(info.filename).name
            if info.is_dir() or not name.endswith(suffix) or name.startswith("."):
                continue
            if name in seen:
                raise ValueError(f"{zip_path} holds two members named {name}")
            seen.add(name)
            target = dest_dir / name
            with archive.open(info) as source, target.open("wb") as out:
                while chunk := source.read(CHUNK_BYTES):
                    out.write(chunk)
    return len(seen)


def clips_dir(dataset_key: str = "neuralink") -> Path:
    """Return the directory holding a zip dataset's extracted clips."""
    return raw_dir(dataset_key) / CLIPS_SUBDIR


def _unpack_if_needed(dataset: Dataset) -> None:
    """Extract a single-zip dataset unless the expected file count is present."""
    if dataset.zip_member_count is None:
        return
    target = clips_dir(dataset.key)
    present = len(list(target.glob("*.wav"))) if target.is_dir() else 0
    if present == dataset.zip_member_count:
        return
    zip_path = raw_dir(dataset.key) / dataset.files[0].name
    count = safe_extract_zip(zip_path, target)
    if count != dataset.zip_member_count:
        raise DownloadError(
            f"{zip_path} unpacked to {count} files, expected {dataset.zip_member_count}"
        )
    print(f"  unpacked {count} files into {target}", file=sys.stderr)


def fetch_dataset(key: str, *, show_progress: bool = True) -> bool:
    """Fetch, verify and unpack one dataset.

    Returns:
        ``True`` if every file is present afterwards, ``False`` if the dataset
        needs a manual download (instructions are printed).

    Raises:
        ChecksumError: If a file is present or downloaded but does not verify.
    """
    dataset = get_dataset(key)
    print(f"[{dataset.key}] {dataset.title}", file=sys.stderr)
    complete = True
    for file in dataset.files:
        try:
            status = fetch_file(dataset, file, show_progress=show_progress)
        except ChecksumError:
            raise
        except DownloadError as exc:
            complete = False
            print(f"  {file.name}: missing ({exc})", file=sys.stderr)
            continue
        note = "" if status.state == "ok" else f" (no registered hash; sha256={status.sha256})"
        print(f"  {file.name}: {status.state}{note}", file=sys.stderr)
    if complete:
        _unpack_if_needed(dataset)
    elif dataset.manual_steps:
        print(f"  Put the files in: {raw_dir(dataset.key)}", file=sys.stderr)
        print(f"  Landing page:     {dataset.landing_url}", file=sys.stderr)
        print("  " + dataset.manual_steps, file=sys.stderr)
    return complete


def print_status() -> None:
    """Print a table of every registered file and its verification state."""
    for dataset in DATASETS.values():
        print(f"{dataset.key}  ({dataset.licence}; used in {dataset.used_in})")
        for file in dataset.files:
            status = file_status(dataset, file)
            detail = f"  sha256={status.sha256}" if status.state == "unverified" else ""
            print(f"  {status.state:<10} {file.name}{detail}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``neurochain-data`` command line. Returns a process exit code."""
    parser = argparse.ArgumentParser(prog="neurochain-data", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="download, verify and unpack datasets")
    fetch.add_argument("keys", nargs="*", help=f"any of {sorted(DATASETS)}; default: all")
    fetch.add_argument(
        "--strict", action="store_true", help="exit non-zero if any dataset is incomplete"
    )
    sub.add_parser("status", help="show what is on disk")
    args = parser.parse_args(argv)

    if args.command == "status":
        print_status()
        return 0

    keys: list[str] = list(args.keys) or list(DATASETS)
    unknown = [key for key in keys if key not in DATASETS]
    if unknown:
        parser.error(f"unknown dataset(s) {unknown}; choose from {sorted(DATASETS)}")
    results = {key: fetch_dataset(key) for key in keys}
    incomplete = [key for key, ok in results.items() if not ok]
    if incomplete:
        print(f"\nNeeds a manual download: {', '.join(incomplete)}", file=sys.stderr)
    return 1 if (incomplete and args.strict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
