#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

# ----------------------------- Utilities -----------------------------

def which_or_die(cmd: str):
    if shutil.which(cmd) is None:
        sys.stderr.write(f"ERROR: required command not found: {cmd}\n")
        sys.exit(1)

def run(cmd: List[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        check=check,
        text=True if capture else False,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )

def remote_join(remote_root: str, *parts: str) -> str:
    base = remote_root.rstrip("/")
    suffix = "/".join(p.strip("/").replace("\\", "/") for p in parts if p)
    return f"{base}/{suffix}" if suffix else base

def ensure_remote_dir(path: str):
    run(["rclone", "mkdir", path], check=True)

def rclone_deletefile(path: str):
    subprocess.run(["rclone", "deletefile", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def rclone_exists(remote_file: str) -> bool:
    parent = os.path.dirname(remote_file)
    name = os.path.basename(remote_file)
    try:
        cp = run(["rclone", "lsjson", parent], check=True, capture=True)
        items = json.loads(cp.stdout or "[]")
        for it in items:
            if it.get("Name") == name or os.path.basename(it.get("Path", "")) == name:
                return True
        return False
    except subprocess.CalledProcessError:
        return False

def choose_remote(remotes: List[str], rel_path: str) -> str:
    h = hashlib.md5(rel_path.encode("utf-8")).hexdigest()
    idx = int(h, 16) % len(remotes)
    return remotes[idx]

def build_paths(remote_root: str, rel_zst: str) -> Tuple[str, str]:
    final = remote_join(remote_root, rel_zst)
    tmp   = remote_join(remote_root, ".incomplete", f"{rel_zst}.partial")
    return final, tmp

# ----------------------------- Proc registry -----------------------------

class ProcRegistry:
    """Tracks child procs so we can nuke them on Ctrl-C."""
    def __init__(self):
        self._lock = threading.Lock()
        self._procs: List[subprocess.Popen] = []

    def add(self, *procs: subprocess.Popen):
        with self._lock:
            self._procs.extend(p for p in procs if p is not None)

    def discard(self, *procs: subprocess.Popen):
        with self._lock:
            for p in procs:
                try:
                    self._procs.remove(p)
                except ValueError:
                    pass

    def kill_all(self):
        with self._lock:
            procs = list(self._procs)
        # First try SIGTERM to each process group, then SIGKILL if needed
        for p in procs:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except Exception:
                pass
        time.sleep(0.4)
        for p in procs:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except Exception:
                    pass
        for p in procs:
            try:
                p.wait(timeout=0.5)
            except Exception:
                pass
        with self._lock:
            self._procs.clear()

PROC_REG = ProcRegistry()

# ----------------------------- Streaming -----------------------------

def stream_compress_to_remote(
    src_file: str,
    tmp_remote: str,
    zstd_threads: int,
    zstd_level: int,
    rclone_progress: bool,
    abort_event: threading.Event,
) -> Tuple[int, int]:
    """
    zstd -<level> -T<threads> -c -- src | rclone rcat tmp
    Returns (rc_zstd, rc_rcat). If abort_event is set, kills procs and returns non-zero.
    """
    zstd_cmd = ["zstd", f"-{zstd_level}", f"-T{zstd_threads}", "-c", "--", src_file]
    rcat_cmd = ["rclone", "rcat", tmp_remote]
    if rclone_progress:
        rcat_cmd.append("--progress")

    zstd = subprocess.Popen(
        zstd_cmd, stdout=subprocess.PIPE, start_new_session=True
    )
    rcat = subprocess.Popen(
        rcat_cmd, stdin=zstd.stdout, start_new_session=True
    )
    PROC_REG.add(zstd, rcat)
    if zstd.stdout:
        zstd.stdout.close()

    # Poll so we can react quickly to aborts
    try:
        while True:
            if abort_event.is_set():
                # kill both process groups
                try: os.killpg(rcat.pid, signal.SIGTERM)
                except Exception: pass
                try: os.killpg(zstd.pid, signal.SIGTERM)
                except Exception: pass
                time.sleep(0.2)
                # hard kill if needed
                if rcat.poll() is None:
                    try: os.killpg(rcat.pid, signal.SIGKILL)
                    except Exception: pass
                if zstd.poll() is None:
                    try: os.killpg(zstd.pid, signal.SIGKILL)
                    except Exception: pass
                # ensure they exit
                try: rcat.wait(timeout=1.0)
                except Exception: pass
                try: zstd.wait(timeout=1.0)
                except Exception: pass
                return (1, 1)

            rc_rcat = rcat.poll()
            rc_zstd = zstd.poll()
            if rc_rcat is not None and rc_zstd is not None:
                return (rc_zstd, rc_rcat)
            time.sleep(0.1)
    finally:
        PROC_REG.discard(zstd, rcat)

# ----------------------------- Data -----------------------------

@dataclass
class TransferTask:
    src: str
    rel: str
    size: int
    attempts: int = 0

@dataclass
class Stats:
    total_files: int = 0
    total_bytes: int = 0
    ok: int = 0
    skipped: int = 0
    requeued: int = 0
    failed: List[str] = field(default_factory=list)
    inflight: int = 0
    bytes_done: int = 0
    start_time: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def inc(self, key: str, amount: int = 1):
        with self.lock:
            setattr(self, key, getattr(self, key) + amount)

    def mark_ok(self, size: int):
        with self.lock:
            self.ok += 1
            self.bytes_done += size

    def snapshot(self):
        with self.lock:
            return {
                "total_files": self.total_files,
                "total_bytes": self.total_bytes,
                "ok": self.ok,
                "skipped": self.skipped,
                "failed_n": len(self.failed),
                "requeued": self.requeued,
                "inflight": self.inflight,
                "bytes_done": self.bytes_done,
                "start_time": self.start_time,
            }

# ----------------------------- Worker -----------------------------

def worker_loop(
    q: "queue.Queue[TransferTask]",
    remotes: List[str],
    zstd_threads: int,
    zstd_level: int,
    max_retries: int,
    rclone_progress: bool,
    stats: Stats,
    abort_event: threading.Event,
):
    while not abort_event.is_set():
        try:
            task: TransferTask = q.get(timeout=0.2)
        except queue.Empty:
            continue

        # If abort requested after dequeuing, mark done and bail
        if abort_event.is_set():
            q.task_done()
            break

        stats.inc("inflight", +1)
        rel_zst = f"{task.rel}.zst"
        remote_root = choose_remote(remotes, task.rel)
        final, tmp = build_paths(remote_root, rel_zst)

        try:
            # Skip if already published
            if rclone_exists(final):
                print(f"[SKIP] {final}")
                stats.inc("skipped")
                continue

            ensure_remote_dir(os.path.dirname(final))
            ensure_remote_dir(os.path.dirname(tmp))
            rclone_deletefile(tmp)  # clean stale partial

            print(f"[XFER] {task.src} -> {final} (try {task.attempts+1})")
            rc_zstd, rc_rcat = stream_compress_to_remote(
                task.src, tmp, zstd_threads, zstd_level, rclone_progress, abort_event
            )

            if abort_event.is_set():
                # Aborting: ensure temp is gone and DO NOT requeue
                rclone_deletefile(tmp)
                print(f"[ABORTED] Killed in-flight: {task.src}")
                return  # exit worker immediately

            if rc_zstd == 0 and rc_rcat == 0:
                mv = subprocess.run(["rclone", "moveto", tmp, final])
                if mv.returncode != 0 and not rclone_exists(final):
                    print(f"[ERR] moveto failed rc={mv.returncode} for {final}")
                    rclone_deletefile(tmp)
                    raise RuntimeError("moveto failed")
                print(f"[OK] {final}")
                stats.mark_ok(task.size)
            else:
                print(f"[ERR] stream failed (zstd={rc_zstd}, rcat={rc_rcat}) for {task.src}")
                rclone_deletefile(tmp)
                raise RuntimeError("stream failed")

        except Exception as e:
            if abort_event.is_set():
                # No requeue on abort
                rclone_deletefile(tmp)
            else:
                task.attempts += 1
                if task.attempts <= max_retries:
                    delay = min(60.0, 1.5 ** task.attempts)
                    print(f"[RETRY] {task.src} in {delay:.1f}s ({task.attempts}/{max_retries}) due to: {e}")
                    time.sleep(delay)
                    q.put(task)
                    stats.inc("requeued")
                else:
                    print(f"[FAIL] {task.src} after {task.attempts} attempts: {e}")
                    with stats.lock:
                        stats.failed.append(task.src)
        finally:
            q.task_done()
            stats.inc("inflight", -1)

# ----------------------------- Progress -----------------------------

def human_bytes(n: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    i = 0
    while n >= 1024 and i < len(units)-1:
        n /= 1024.0
        i += 1
    return f"{n:.1f} {units[i]}"

def human_time(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def progress_loop(q: "queue.Queue", stats: Stats, interval: float, stop_event: threading.Event):
    while not stop_event.is_set():
        time.sleep(interval)
        snap = stats.snapshot()
        elapsed = max(1e-3, time.time() - snap["start_time"])
        done = snap["ok"] + snap["skipped"] + snap["failed_n"]
        rate_fph = (done / elapsed) * 60
        rate_bps = snap["bytes_done"] / elapsed
        eta_sec = (snap["total_bytes"] - snap["bytes_done"]) / rate_bps if rate_bps > 0 else 0
        print(
            f"[STATUS {time.strftime('%H:%M:%S')}] "
            f"total={snap['total_files']} | done={done} "
            f"(ok={snap['ok']}, skip={snap['skipped']}, fail={snap['failed_n']}) | "
            f"inflight={snap['inflight']} | queued~{q.qsize()} | "
            f"bytes={human_bytes(snap['bytes_done'])}/{human_bytes(snap['total_bytes'])} | "
            f"rate={rate_fph:.2f} f/h, {human_bytes(rate_bps)}/s | ETA~{human_time(eta_sec)}"
        )

# ----------------------------- Discovery & setup -----------------------------

def build_task_list(source: Path) -> List[TransferTask]:
    tasks: List[TransferTask] = []
    src_root = source.resolve()
    for root, _, files in os.walk(src_root):
        for fn in files:
            if fn.lower().endswith(".fli"):
                abs_path = Path(root, fn).resolve()
                rel = str(abs_path.relative_to(src_root)).replace("\\", "/")
                try:
                    size = abs_path.stat().st_size
                except OSError:
                    size = 0
                tasks.append(TransferTask(str(abs_path), rel, size))
    return tasks

def setup_remotes(remotes: List[str]):
    for r in remotes:
        run(["rclone", "mkdir", r], check=True)
        run(["rclone", "mkdir", remote_join(r, ".incomplete")], check=True)

# ----------------------------- CLI -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stripe-compressed .fli transfers across multiple rclone remotes with atomic publish, retries, periodic progress, and immediate Ctrl-C abort."
    )
    p.add_argument("--source", required=True, help="Source directory (recursive).")
    p.add_argument("--remotes", required=True, help="Comma-separated rclone remote roots.")
    p.add_argument("--workers", type=int, default=20, help="Concurrent file transfers.")
    p.add_argument("--zstd-threads", type=int, default=24, help="zstd threads per file (-T).")
    p.add_argument("--zstd-level", type=int, default=10, help="zstd compression level.")
    p.add_argument("--max-retries", type=int, default=3, help="Max requeues per file.")
    p.add_argument("--status-interval", type=float, default=10.0, help="Seconds between status updates.")
    p.add_argument("--rclone-progress", action="store_true", help="Show rclone per-file progress (verbose).")
    return p.parse_args()

# ----------------------------- Main -----------------------------

def main():
    for cmd in ("rclone", "zstd"):
        which_or_die(cmd)

    args = parse_args()
    source = Path(args.source)
    if not source.is_dir():
        sys.stderr.write(f"ERROR: --source not a directory: {source}\n")
        sys.exit(2)

    remotes = [r.strip() for r in args.remotes.split(",") if r.strip()]
    if not remotes:
        sys.stderr.write("ERROR: --remotes is empty.\n")
        sys.exit(2)

    os.environ["LC_ALL"] = "C"

    print("Preparing remotes…")
    setup_remotes(remotes)

    tasks = build_task_list(source)
    if not tasks:
        print("No .fli files found. Nothing to do.")
        return

    stats = Stats(
        total_files=len(tasks),
        total_bytes=sum(t.size for t in tasks),
        start_time=time.time(),
    )

    q: "queue.Queue[TransferTask]" = queue.Queue()
    for t in tasks:
        q.put(t)

    abort_event = threading.Event()
    progress_stop = threading.Event()

    def drain_queue():
        # Clear remaining tasks so nothing else starts
        drained = 0
        try:
            while True:
                _ = q.get_nowait()
                q.task_done()
                drained += 1
        except queue.Empty:
            pass
        if drained:
            print(f"[ABORT] Drained {drained} queued task(s).")

    def handle_signal(sig, frame):
        print("\n[ABORT] Ctrl-C received: killing in-flight transfers and cancelling remaining…")
        abort_event.set()
        progress_stop.set()
        PROC_REG.kill_all()
        drain_queue()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # start workers
    workers = []
    for _ in range(max(1, args.workers)):
        th = threading.Thread(
            target=worker_loop,
            args=(q, remotes, args.zstd_threads, args.zstd_level, args.max_retries, args.rclone_progress, stats, abort_event),
            daemon=True,
        )
        th.start()
        workers.append(th)

    # progress reporter
    reporter = threading.Thread(target=progress_loop, args=(q, stats, args.status_interval, progress_stop), daemon=True)
    reporter.start()

    # Wait until all tasks are finished (or aborted). If aborted, q.join() will return
    # once in-flight tasks have acknowledged .task_done() and the queue was drained.
    try:
        q.join()
    finally:
        progress_stop.set()
        for th in workers:
            th.join(timeout=1.0)

    # Print summary
    snap = stats.snapshot()
    elapsed = max(1e-3, time.time() - snap["start_time"])
    def human_time(seconds: float) -> str:
        seconds = int(max(0, seconds))
        h, rem = divmod(seconds, 3600); m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    print("\n=== Summary ===")
    print(f"Elapsed:   {human_time(elapsed)}")
    print(f"Files:     total={snap['total_files']} ok={snap['ok']} skip={snap['skipped']} fail={snap['failed_n']}")
    print(f"Bytes:     {human_bytes(snap['bytes_done'])}/{human_bytes(snap['total_bytes'])}")

    # Exit 130 on Ctrl-C (conventional), 1 if any failed otherwise 0
    if abort_event.is_set():
        sys.exit(130)
    sys.exit(0 if snap['failed_n'] == 0 else 1)

if __name__ == "__main__":
    main()