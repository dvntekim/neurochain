import hashlib
import threading
import zipfile
from collections.abc import Iterator
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from neurochain.data import download
from neurochain.data.registry import DATASETS, Dataset, RemoteFile, get_dataset

PAYLOAD = b"neural data " * 1000
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def server(tmp_path: Path) -> Iterator[str]:
    """Serve ``tmp_path/served`` over HTTP on a free local port."""
    served = tmp_path / "served"
    served.mkdir()
    (served / "payload.bin").write_bytes(PAYLOAD)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(served)))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    thread.join()


def make_dataset(urls: tuple[str, ...], sha256: str | None = PAYLOAD_SHA) -> Dataset:
    return Dataset(
        key="demo",
        title="Demo",
        landing_url="https://example.org",
        licence="CC0",
        used_in="tests",
        files=(RemoteFile(name="payload.bin", urls=urls, sha256=sha256),),
        manual_steps="Download it by hand.",
    )


def test_sha256_of(tmp_path: Path) -> None:
    path = tmp_path / "f"
    path.write_bytes(PAYLOAD)
    assert download.sha256_of(path) == PAYLOAD_SHA


def test_fetch_url_writes_verified_file(server: str, tmp_path: Path) -> None:
    dest = tmp_path / "out" / "payload.bin"
    digest = download.fetch_url(f"{server}/payload.bin", dest, expected_sha256=PAYLOAD_SHA)
    assert digest == PAYLOAD_SHA
    assert dest.read_bytes() == PAYLOAD
    assert not dest.with_name("payload.bin.part").exists()


def test_fetch_url_checksum_mismatch_leaves_nothing_behind(server: str, tmp_path: Path) -> None:
    dest = tmp_path / "payload.bin"
    with pytest.raises(download.ChecksumError):
        download.fetch_url(f"{server}/payload.bin", dest, expected_sha256="0" * 64)
    assert not dest.exists()
    assert not dest.with_name("payload.bin.part").exists()


def test_fetch_file_falls_back_to_the_next_url(server: str, data_root: Path) -> None:
    dataset = make_dataset((f"{server}/gone.bin", f"{server}/payload.bin"))
    status = download.fetch_file(dataset, dataset.files[0], show_progress=False)
    assert status.state == "ok"
    assert status.path == data_root / "raw" / "demo" / "payload.bin"


def test_fetch_file_reports_every_failed_url(server: str, data_root: Path) -> None:
    dataset = make_dataset((f"{server}/a.bin", f"{server}/b.bin"))
    with pytest.raises(download.DownloadError, match=r"a\.bin(.|\n)*b\.bin"):
        download.fetch_file(dataset, dataset.files[0], show_progress=False)


def test_file_status_states(data_root: Path) -> None:
    dataset = make_dataset(())
    file = dataset.files[0]
    assert download.file_status(dataset, file).state == "missing"
    path = data_root / "raw" / "demo" / "payload.bin"
    path.parent.mkdir(parents=True)
    path.write_bytes(PAYLOAD)
    assert download.file_status(dataset, file).state == "ok"
    path.write_bytes(b"tampered")
    assert download.file_status(dataset, file).state == "corrupt"
    unpinned = make_dataset((), sha256=None)
    status = download.file_status(unpinned, unpinned.files[0])
    assert status.state == "unverified"
    assert status.sha256 == hashlib.sha256(b"tampered").hexdigest()


def test_corrupt_file_on_disk_is_an_error_not_a_redownload(data_root: Path) -> None:
    dataset = make_dataset(("http://127.0.0.1:9/never-contacted",))
    path = data_root / "raw" / "demo" / "payload.bin"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"tampered")
    with pytest.raises(download.ChecksumError, match="Delete it"):
        download.fetch_file(dataset, dataset.files[0], show_progress=False)


def test_manual_dataset_is_reported_not_raised(
    data_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(DATASETS, "demo", make_dataset(()))
    assert download.fetch_dataset("demo", show_progress=False) is False
    assert "Download it by hand." in capsys.readouterr().err


def test_token_gated_dataset_without_token_is_manual(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DRYAD_TOKEN", raising=False)
    nomad = get_dataset("nomad")
    with pytest.raises(download.DownloadError, match="manual"):
        download.fetch_file(nomad, nomad.files[0], show_progress=False)


def test_safe_extract_zip_flattens_hostile_paths(tmp_path: Path) -> None:
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("data/one.wav", b"1")
        zf.writestr("../../escape.wav", b"2")
        zf.writestr("/abs/three.wav", b"3")
        zf.writestr("data/notes.txt", b"ignored")
        zf.writestr("data/.hidden.wav", b"ignored")
    dest = tmp_path / "clips"
    assert download.safe_extract_zip(archive, dest) == 3
    assert sorted(p.name for p in dest.iterdir()) == ["escape.wav", "one.wav", "three.wav"]
    assert not (tmp_path / "escape.wav").exists()


def test_safe_extract_zip_rejects_duplicate_names(tmp_path: Path) -> None:
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("a/x.wav", b"1")
        zf.writestr("b/x.wav", b"2")
    with pytest.raises(ValueError, match="two members"):
        download.safe_extract_zip(archive, tmp_path / "clips")


def test_zip_dataset_is_unpacked_and_count_checked(
    server: str, tmp_path: Path, data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "served" / "data.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for i in range(3):
            zf.writestr(f"data/{i}.wav", bytes([i]))
    dataset = Dataset(
        key="zipped",
        title="Zipped",
        landing_url="https://example.org",
        licence="CC0",
        used_in="tests",
        files=(
            RemoteFile(
                name="data.zip", urls=(f"{server}/data.zip",), sha256=download.sha256_of(archive)
            ),
        ),
        zip_member_count=3,
    )
    monkeypatch.setitem(DATASETS, "zipped", dataset)
    assert download.fetch_dataset("zipped", show_progress=False) is True
    assert len(list(download.clips_dir("zipped").glob("*.wav"))) == 3
    # A second call finds everything in place and does not need the server.
    assert download.fetch_dataset("zipped", show_progress=False) is True


def test_cli_status_and_unknown_key(data_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert download.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "neuralink" in out and "missing" in out
    with pytest.raises(SystemExit):
        download.main(["fetch", "not-a-dataset"])


def test_registry_lookup() -> None:
    assert get_dataset("neuralink").zip_member_count == 743
    with pytest.raises(KeyError, match="choose from"):
        get_dataset("nope")
    for dataset in DATASETS.values():
        for file in dataset.files:
            assert file.sha256 is None or len(file.sha256) == 64
