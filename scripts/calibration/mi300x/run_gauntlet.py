#!/usr/bin/env python3
"""ml8 gauntlet dispatcher — pin one calibration cell per GPU, chew through the
manifest, resume-aware. Built for an 8×MI300X pod (saturate all 8), but works with
any GPU count (1 for the single-instance validation, 2 locally, etc.).

    python3 run_gauntlet.py --manifest gauntlet_tier1.json --gpus 8 \
        --models-root /models --out-root /gauntlet-out

Behavior:
  - N worker threads (one per GPU). Each pops the next pending job, runs a resident
    calibration pinned to its GPU, writes the blob bundle, marks the job DONE.
  - Resume-aware twice over: a finished job (GAUNTLET_DONE marker) is skipped on
    rerun; an interrupted job resumes from its own per-layer blobs (the calibrator
    omits --no-resume). So a crashed pod just needs the dispatcher rerun.
  - PPL is NOT run here (ml8 inference is gfx1201-only). Output per cell = blobs +
    the calibrator's manifest.json (Y_SNR per linear), which ranks the tiers.

Honest status: the dense cells run today (--resident dense is shipped + tested).
The MoE cells need the resident-MoE path (the 192 GB pod fits 35B resident, but the
no-pager MoE loader isn't built yet) — those jobs will fail loudly until it lands.
"""
import argparse
import json
import queue
import subprocess
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CALIB = HERE.parent / "calibrate_ml8_paged.py"

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_cmd(job: dict, gpu: int, models_root: Path, out_root: Path,
              calib_script: Path) -> list[str]:
    mdir = models_root / job["model_key"]
    out_dir = out_root / job["id"]
    cmd = [
        "python3", str(calib_script),
        "--strategy", job["strategy"], "--resident",
        "--model", str(mdir),
        "--gguf", str(mdir / job["gguf"]),
        "--arch", job["arch"],
        "--rotation", job["rotation"],
        "--snap-centroids", job["snap_centroids"],
        "--fit-loss", job["fit_loss"],
        "--n-centroids", str(job["n_centroids"]),
        "--group-size", str(job["group_size"]),
        "--group-size-down", str(job["group_size_down"]),
        "--n-samples", str(job["n_samples"]),
        "--seq-len", str(job["seq_len"]),
        "--heavy-rounds", str(job["heavy_rounds"]),
        "--heavy-steps", str(job["heavy_steps"]),
        "--heavy-dtype", job["heavy_dtype"],
        "--output-dir", str(out_dir),
        "--device", f"cuda:{gpu}",
    ]
    if job.get("act_order"):
        cmd.append("--act-order")
    return cmd


def worker(gpu: int, q: "queue.Queue[dict]", args, calib_script: Path) -> None:
    models_root = Path(args.models_root)
    out_root = Path(args.out_root)
    while True:
        try:
            job = q.get_nowait()
        except queue.Empty:
            return
        out_dir = out_root / job["id"]
        done_marker = out_dir / "GAUNTLET_DONE"
        if done_marker.exists():
            log(f"GPU{gpu}  SKIP {job['id']} (already done)")
            q.task_done()
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_cmd(job, gpu, models_root, out_root, calib_script)
        log(f"GPU{gpu}  START {job['id']}  (bpv={job['avg_bpv']}, {job['role']})")
        if args.dry_run:
            log(f"GPU{gpu}  DRY  {' '.join(cmd)}")
            q.task_done()
            continue
        t0 = time.time()
        with open(out_dir / "run.log", "w") as lf:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
        dt = (time.time() - t0) / 60
        if rc == 0:
            done_marker.write_text(f"ok {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
            log(f"GPU{gpu}  DONE  {job['id']}  ({dt:.1f} min)")
        else:
            log(f"GPU{gpu}  FAIL  {job['id']}  rc={rc} ({dt:.1f} min) — see {out_dir/'run.log'}")
        q.task_done()


def summarize(manifest: dict, out_root: Path) -> None:
    log("==== gauntlet summary (Y_SNR mean over linears; PPL on R9700 later) ====")
    rows = []
    for job in manifest["jobs"]:
        mpath = Path(out_root) / job["id"] / "manifest.json"
        ysnr = None
        if mpath.exists():
            try:
                res = json.load(open(mpath)).get("results", [])
                ys = [r["y_snr_db"] for r in res if "y_snr_db" in r]
                ysnr = sum(ys) / len(ys) if ys else None
            except Exception:
                pass
        rows.append((job["id"], job["avg_bpv"], ysnr))
    rows.sort(key=lambda r: (r[2] is None, -(r[2] or 0)))
    for jid, bpv, ysnr in rows:
        ys = f"{ysnr:.2f} dB" if ysnr is not None else "  (no result)"
        with _print_lock:
            print(f"  {jid:30} bpv={bpv:<6} Y_SNR={ys}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--gpus", default="1",
                    help="GPU count (e.g. 8) or explicit list (e.g. 0,1,2,3)")
    ap.add_argument("--models-root", default="/models")
    ap.add_argument("--out-root", default="/gauntlet-out")
    ap.add_argument("--calib-script", default=str(DEFAULT_CALIB))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    gpus = ([int(x) for x in args.gpus.split(",")] if "," in args.gpus
            else list(range(int(args.gpus))))
    manifest = json.load(open(args.manifest))
    jobs = manifest["jobs"]
    calib_script = Path(args.calib_script)

    log(f"gauntlet '{manifest.get('name')}': {len(jobs)} jobs over {len(gpus)} GPU(s) {gpus}")
    q: "queue.Queue[dict]" = queue.Queue()
    for j in jobs:
        q.put(j)

    threads = [threading.Thread(target=worker, args=(g, q, args, calib_script), daemon=False)
               for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    summarize(manifest, Path(args.out_root))
    log("gauntlet complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
