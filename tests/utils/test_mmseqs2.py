"""Tests for run_mmseqs2 against a local stand-in for the MMseqs2 API server."""

from __future__ import annotations

import io
import json
import tarfile
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from sampleworks.utils import mmseqs2
from sampleworks.utils.mmseqs2 import run_mmseqs2


QUERY_SEQUENCE = "MKTAYIAKQR"
HOMOLOG_SEQUENCE = "MKTAYLAKQR"


def _result_archive() -> bytes:
    """Build the tar.gz the server returns for an unpaired ``env`` job on one sequence."""
    a3m = f">101\n{QUERY_SEQUENCE}\n>homolog\n{HOMOLOG_SEQUENCE}\n".encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name in ("uniref.a3m", "bfd.mgnify30.metaeuk30.smag30.a3m"):
            info = tarfile.TarInfo(name)
            info.size = len(a3m)
            tar.addfile(info, io.BytesIO(a3m))
    return buffer.getvalue()


@dataclass
class FakeMSAServer:
    """MMseqs2 API stand-in whose first ``failed_downloads`` downloads fail.

    Parameters
    ----------
    url : str
        Base URL the server listens on.
    failed_downloads : int
        How many download requests to answer with ``failure_status`` and
        ``failure_body`` before answering with a valid archive.
    failure_status : int
        HTTP status code of a failed download.
    failure_body : bytes
        Response body of a failed download.
    download_requests : int
        Number of download requests received so far.
    """

    url: str
    failed_downloads: int = 0
    failure_status: int = 404
    failure_body: bytes = b"Not Found"
    download_requests: int = field(default=0)


@pytest.fixture
def msa_server(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeMSAServer]:
    """Serve the MMseqs2 ticket and download endpoints on localhost.

    Side effects: sets ``NO_PROXY`` and zeroes ``RETRY_WAIT_SECONDS`` for the test.
    """
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setattr(mmseqs2, "RETRY_WAIT_SECONDS", 0)
    archive = _result_archive()
    state = FakeMSAServer(url="")

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status: int, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            self.rfile.read(int(self.headers["Content-Length"]))
            self._reply(200, json.dumps({"status": "COMPLETE", "id": "job-1"}).encode())

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/ticket/job-1":
                self._reply(200, json.dumps({"status": "COMPLETE", "id": "job-1"}).encode())
            elif self.path == "/result/download/job-1":
                state.download_requests += 1
                if state.download_requests <= state.failed_downloads:
                    self._reply(state.failure_status, state.failure_body)
                else:
                    self._reply(200, archive)
            else:
                self._reply(404, b"Not Found")

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state.url = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield state
    server.shutdown()
    server.server_close()
    thread.join()


def _run(msa_server: FakeMSAServer, prefix: Path) -> list[str]:
    return run_mmseqs2(QUERY_SEQUENCE, str(prefix), use_env=True, host_url=msa_server.url)


def test_successful_fetch_returns_alignment_and_caches_archive(
    msa_server: FakeMSAServer, tmp_path: Path
):
    prefix = tmp_path / "target_unpaired_tmp"

    alignments = _run(msa_server, prefix)

    assert len(alignments) == 1
    assert QUERY_SEQUENCE in alignments[0] and HOMOLOG_SEQUENCE in alignments[0]
    assert tarfile.is_tarfile(tmp_path / "target_unpaired_tmp_env" / "out.tar.gz")


@pytest.mark.parametrize(
    ("status", "body"),
    [(404, b"Not Found"), (200, b"<html>Service unavailable</html>")],
    ids=["http-404", "non-archive-body"],
)
def test_failed_download_raises_and_leaves_no_cache_directory(
    msa_server: FakeMSAServer, tmp_path: Path, status: int, body: bytes
):
    msa_server.failed_downloads = 100
    msa_server.failure_status = status
    msa_server.failure_body = body
    prefix = tmp_path / "target_unpaired_tmp"

    with pytest.raises(Exception, match="Too many failed attempts"):
        _run(msa_server, prefix)

    assert list(tmp_path.iterdir()) == []


def test_transient_download_failure_is_retried(msa_server: FakeMSAServer, tmp_path: Path):
    msa_server.failed_downloads = 2

    alignments = _run(msa_server, tmp_path / "target_unpaired_tmp")

    assert msa_server.download_requests == 3
    assert HOMOLOG_SEQUENCE in alignments[0]


def test_fetch_succeeds_after_an_earlier_failed_fetch(msa_server: FakeMSAServer, tmp_path: Path):
    msa_server.failed_downloads = 6
    prefix = tmp_path / "target_unpaired_tmp"
    with pytest.raises(Exception, match="Too many failed attempts"):
        _run(msa_server, prefix)

    alignments = _run(msa_server, prefix)

    assert HOMOLOG_SEQUENCE in alignments[0]


def test_invalid_cached_archive_is_discarded_and_fetched_again(
    msa_server: FakeMSAServer, tmp_path: Path
):
    # What a failed download wrote before downloads were validated.
    cache_dir = tmp_path / "target_unpaired_tmp_env"
    cache_dir.mkdir()
    (cache_dir / "out.tar.gz").write_bytes(b"Not Found")

    alignments = _run(msa_server, tmp_path / "target_unpaired_tmp")

    assert msa_server.download_requests == 1
    assert HOMOLOG_SEQUENCE in alignments[0]
    assert tarfile.is_tarfile(cache_dir / "out.tar.gz")
