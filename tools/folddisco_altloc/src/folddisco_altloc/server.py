"""Local web interface: upload a structure, pick a loop and options, see the summary.

A small standard-library HTTP server. It listens on localhost by default and keeps
uploads and runs under a work directory on disk; the browser page talks to it
through a JSON API:

    POST /api/upload            {filename, content}       -> structure summary
    POST /api/run               {upload_id, chain, ...}   -> {job_id}
    GET  /api/jobs/<id>                                   -> progress and results
    GET  /api/jobs/<id>/files.zip                         -> every file of the run
    GET  /api/jobs/<id>/summary.csv                       -> the summary table
"""

import errno
import io
import json
import re
import threading
import uuid
import webbrowser
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path

from .loop_motif import find_altloc_segments, MotifError
from .pipeline import (
    MotifOptions,
    prepare_motif,
    PreparedMotif,
    run_search,
    SearchOptions,
    SearchOutcome,
)
from .query import AA_SUBST_MODES, DEFAULT_DATABASES, RESULT_PAGE_URL
from .summary import to_csv, totals


MAX_UPLOAD_BYTES = 50 * 1024 * 1024
PDB_SUFFIXES = (".pdb", ".ent")


@dataclass
class Upload:
    """A structure uploaded through the page."""

    path: Path
    chains: list[str]


@dataclass
class Job:
    """One run started from the page; mutated by its worker thread."""

    run_dir: Path
    state: str = "preparing"
    messages: list[str] = field(default_factory=list)
    error: str | None = None
    prepared: PreparedMotif | None = None
    outcome: SearchOutcome | None = None


class AppState:
    """Uploads and jobs of one server process, guarded by a lock."""

    def __init__(self, workdir: Path) -> None:
        self.workdir = workdir
        self.uploads: dict[str, Upload] = {}
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()


def _safe_filename(name: str) -> str:
    """Basename with anything but letters, digits, ``._-`` replaced by ``_``."""
    return re.sub(r"[^\w.-]", "_", Path(name).name) or "structure.pdb"


def handle_upload(state: AppState, payload: dict) -> dict:
    """Save an uploaded PDB and describe its chains and altloc segments.

    Parameters
    ----------
    state : AppState
        Server state.
    payload : dict
        ``{"filename": str, "content": str}``.

    Returns
    -------
    dict
        ``upload_id``, ``filename``, ``chains`` and ``segments`` for the page.

    Raises
    ------
    MotifError
        If the file is not a PDB-format structure with atoms.
    """
    filename = _safe_filename(str(payload.get("filename", "")))
    content = str(payload.get("content", ""))
    if not filename.lower().endswith(PDB_SUFFIXES):
        raise MotifError(
            "only PDB-format files (.pdb, .ent) are supported; convert mmCIF to PDB first"
        )
    lines = content.splitlines(keepends=True)
    atom_lines = [line for line in lines if line.startswith(("ATOM  ", "HETATM"))]
    if not atom_lines:
        raise MotifError(f"{filename} has no ATOM or HETATM records")

    upload_id = uuid.uuid4().hex[:12]
    path = state.workdir / "uploads" / upload_id / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    chains = sorted({line[21] for line in atom_lines if line.startswith("ATOM  ")})
    with state.lock:
        state.uploads[upload_id] = Upload(path, chains)

    segments = [
        {
            "chain": s.chain,
            "residues": s.residues,
            "length": s.end - s.start + 1,
            "altlocs": list(s.altlocs),
            "incomplete": [r.label for r in s.incomplete],
        }
        for s in find_altloc_segments(lines)
    ]
    return {"upload_id": upload_id, "filename": filename, "chains": chains, "segments": segments}


def _options_from(payload: dict) -> tuple[MotifOptions, SearchOptions]:
    """Motif and search options from a ``/api/run`` request body."""
    altlocs = payload.get("altlocs") or None
    aa_subst = payload.get("aa_subst") or None
    if aa_subst is not None and aa_subst not in AA_SUBST_MODES:
        raise MotifError(f"aa_subst must be one of {', '.join(AA_SUBST_MODES)}")
    databases = tuple(payload.get("databases") or DEFAULT_DATABASES)
    unknown = set(databases) - set(DEFAULT_DATABASES)
    if unknown:
        raise MotifError(f"unknown databases: {', '.join(sorted(unknown))}")
    min_matched = payload.get("min_matched")
    motif = MotifOptions(
        motif_size=int(payload.get("motif_size", 4)),
        cutoff=float(payload.get("cutoff", 12.0)),
        min_separation=int(payload.get("min_separation", 3)),
        altlocs=(str(altlocs[0]), str(altlocs[1])) if altlocs else None,
    )
    search = SearchOptions(
        aa_subst=aa_subst,
        confident=bool(payload.get("confident", False)),
        rmsd_max=float(payload.get("rmsd_max", 0.5)),
        min_matched=int(min_matched) if min_matched else None,
        databases=databases,
    )
    return motif, search


def _work(
    job: Job,
    upload: Upload,
    chain: str,
    residues: str,
    motif_options: MotifOptions,
    search_options: SearchOptions,
) -> None:
    """Worker thread: prepare the motif, then search; records progress on ``job``."""
    try:
        job.prepared = prepare_motif(upload.path, chain, residues, motif_options, job.run_dir)
        job.messages.append(f"motif {job.prepared.choice.motif}")
        job.state = "searching"
        job.outcome = run_search(job.prepared, search_options, on_message=job.messages.append)
        job.state = "done"
    except Exception as error:  # reported to the page; the server keeps running
        job.error = str(error) or type(error).__name__
        job.state = "error"


def handle_run(state: AppState, payload: dict) -> dict:
    """Start a run in a background thread.

    Parameters
    ----------
    state : AppState
        Server state.
    payload : dict
        ``upload_id``, ``chain``, ``residues`` and the options read by
        :func:`_options_from`.

    Returns
    -------
    dict
        ``{"job_id": str}``.
    """
    with state.lock:
        upload = state.uploads.get(str(payload.get("upload_id")))
    if upload is None:
        raise MotifError("upload not found; upload the structure again")
    chain = str(payload.get("chain", "")).strip()
    residues = str(payload.get("residues", "")).replace(" ", "")
    if not chain or not residues:
        raise MotifError("choose a chain and residues")
    motif_options, search_options = _options_from(payload)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = (
        state.workdir / "runs" / f"{stamp}_{upload.path.stem}_{chain}{residues.replace(',', '_')}"
    )
    job_id = uuid.uuid4().hex[:12]
    job = Job(run_dir)
    with state.lock:
        state.jobs[job_id] = job
    threading.Thread(
        target=_work,
        args=(job, upload, chain, residues, motif_options, search_options),
        daemon=True,
    ).start()
    return {"job_id": job_id}


def job_view(job: Job) -> dict:
    """JSON-ready view of a job's progress and results for the page.

    Parameters
    ----------
    job : Job
        The job.

    Returns
    -------
    dict
        State, messages, motif (once chosen) and summary table (once searched).
    """
    view: dict = {
        "state": job.state,
        "messages": list(job.messages),
        "error": job.error,
        "run_dir": str(job.run_dir),
    }
    prepared = job.prepared
    if prepared is not None:
        choice = prepared.choice
        view["motif"] = {
            "motif": choice.motif,
            "altlocs": prepared.altlocs,
            "moving": [
                {"label": r.label, "score": round(choice.loop_scores[r], 1)} for r in choice.loop
            ],
            "anchors": [
                {"label": r.label, "score": round(choice.anchor_scores[r], 1)}
                for r in choice.anchors
            ],
            "distances": [
                [
                    [round(d, 1) for d in prepared.cb_distances[(moving, anchor)]]
                    for anchor in choice.anchors
                ]
                for moving in choice.loop
            ],
        }
    outcome = job.outcome
    if outcome is not None:
        f = outcome.hit_filter
        view["search"] = {
            "submitted_motif": outcome.submitted_motif,
            "pages": {a: f"{RESULT_PAGE_URL}/{t}" for a, t in outcome.tickets.items()},
            "filter": {
                "motif_size": f.motif_size,
                "min_matched": f.min_matched,
                "rmsd_max": f.rmsd_max,
                "confident": f.confident,
            },
            "rows": [
                {
                    "database": row.database,
                    "shared": row.shared,
                    "counts": {
                        a: {
                            "targets": len(c.targets),
                            "numberings": c.numberings,
                            "returned": c.returned,
                            "last_rank": c.last_rank,
                            "possibly_truncated": c.possibly_truncated,
                        }
                        for a, c in row.counts.items()
                    },
                }
                for row in outcome.rows
            ],
            "totals": totals(outcome.rows),
        }
    return view


def zip_directory(directory: Path) -> bytes:
    """Zip every file under ``directory`` (paths relative to it)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                archive.write(path, f"{directory.name}/{path.relative_to(directory)}")
    return buffer.getvalue()


def make_handler(state: AppState) -> type[BaseHTTPRequestHandler]:
    """Request handler class bound to ``state``."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass  # keep the terminal quiet; errors reach the page

        def _send(
            self, status: int, body: bytes, content_type: str, filename: str | None = None
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, data: dict) -> None:
            self._send(status, json.dumps(data).encode(), "application/json")

        def _job(self, job_id: str) -> Job | None:
            with state.lock:
                return state.jobs.get(job_id)

        def do_GET(self) -> None:  # noqa: N802
            parts = [p for p in self.path.split("?")[0].split("/") if p]
            if not parts:
                page = files("folddisco_altloc").joinpath("static/index.html").read_bytes()
                self._send(HTTPStatus.OK, page, "text/html; charset=utf-8")
            elif parts[:2] == ["api", "jobs"] and len(parts) >= 3 and (job := self._job(parts[2])):
                if len(parts) == 3:
                    self._json(HTTPStatus.OK, job_view(job))
                elif parts[3] == "files.zip" and job.run_dir.exists():
                    self._send(
                        HTTPStatus.OK,
                        zip_directory(job.run_dir),
                        "application/zip",
                        f"{job.run_dir.name}.zip",
                    )
                elif parts[3] == "summary.csv" and job.outcome is not None:
                    self._send(
                        HTTPStatus.OK,
                        to_csv(job.outcome.rows).encode(),
                        "text/csv",
                        f"{job.run_dir.name}.summary.csv",
                    )
                else:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "not available yet"})
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            if length > MAX_UPLOAD_BYTES:
                self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "file larger than 50 MB"})
                return
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/api/upload":
                    self._json(HTTPStatus.OK, handle_upload(state, payload))
                elif self.path == "/api/run":
                    self._json(HTTPStatus.OK, handle_run(state, payload))
                else:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            except (MotifError, ValueError, TypeError, KeyError) as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

    return Handler


def bind_server(host: str, port: int, state: AppState, attempts: int = 20) -> ThreadingHTTPServer:
    """Bind to ``port``, or to the next free port if it is taken.

    Parameters
    ----------
    host : str
        Address to listen on.
    port : int
        First port to try.
    state : AppState
        Server state for the request handler.
    attempts : int, optional
        How many consecutive ports to try.

    Returns
    -------
    ThreadingHTTPServer
        Bound, not yet serving.

    Raises
    ------
    OSError
        If none of the ports is free.
    """
    handler = make_handler(state)
    for candidate in range(port, port + attempts):
        try:
            return ThreadingHTTPServer((host, candidate), handler)
        except OSError as error:
            if error.errno != errno.EADDRINUSE or candidate == port + attempts - 1:
                raise
    raise AssertionError("unreachable")


def serve(host: str, port: int, workdir: Path, open_browser: bool = True) -> None:
    """Run the web interface until interrupted.

    Parameters
    ----------
    host : str
        Address to listen on; keep ``127.0.0.1`` unless others should reach it.
    port : int
        Preferred port; the next free one is used if it is taken.
    workdir : Path
        Where uploads and runs are saved.
    open_browser : bool, optional
        Open the page in the default browser.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    server = bind_server(host, port, AppState(workdir))
    port = server.server_address[1]
    url = f"http://{'localhost' if host in ('127.0.0.1', '::1') else host}:{port}/"
    print(f"folddisco-altloc: {url}  (runs saved under {workdir}; Ctrl-C to stop)", flush=True)
    if host not in ("127.0.0.1", "localhost", "::1"):
        print("warning: listening beyond this machine; anyone who can reach it can submit searches")
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
