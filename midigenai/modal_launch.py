"""
Durable launcher for long Modal training runs.

`modal run --detach` cancels the app a minute or two after the local client
exits for ANY reason (laptop sleep killed medium_full_v1 twice, see the
PLAN.md log). The pattern that survives: deploy the app, then spawn the
training function from a short-lived client and walk away.

    # 1. deploy (re-run after any code change; picks up the local midigenai/)
    python -m midigenai.modal_launch deploy

    # 2. spawn a run; prints the function-call id to keep
    python -m midigenai.modal_launch spawn --run-name v4_full --corpus corpus_full_v4_24 \
        --size medium --batch-size 64 --grad-accum 1 --block-size 2048 \
        --max-steps 180000 --lr 4e-4 --schedule wsd --compile --stage-local \
        --mixture aria:0.5

    # 3. check on it (status + last metrics rows from the runs volume)
    python -m midigenai.modal_launch status --run-name v4_full [--call-id fc-...]

Health rule from the log: believe a run is alive only when the step counter
advances between two timestamped reads, never from a single snapshot.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

APP_NAME = "midigenai-train"
RUNS_VOLUME = "midigenai-runs"


def deploy() -> None:
    here = Path(__file__).resolve().parent
    subprocess.run([sys.executable, "-m", "modal", "deploy", str(here / "modal_train.py")],
                   check=True)


def spawn(args: argparse.Namespace) -> str:
    import modal
    fn = modal.Function.from_name(APP_NAME, "train")
    kwargs = dict(
        size=args.size, max_steps=args.max_steps, batch_size=args.batch_size,
        grad_accum=args.grad_accum, block_size=args.block_size,
        warmup_steps=args.warmup_steps, eval_interval=args.eval_interval,
        save_interval=args.save_interval, lr=args.lr, schedule=args.schedule,
        decay_steps=args.decay_steps, augment=not args.no_augment,
        aug_pitch=args.aug_pitch, aug_velocity=args.aug_velocity,
        doc_start_frac=args.doc_start_frac, mixture=args.mixture,
        corpus=args.corpus, compile=args.compile, stage_local=args.stage_local,
        run_name=args.run_name, resume=args.resume, resume_from=args.resume_from,
        header_dropout=args.header_dropout, header_drop_all=args.header_drop_all,
        rope_base=args.rope_base,
    )
    call = fn.spawn(**kwargs)
    print(f"spawned {args.run_name}: call id {call.object_id}")
    print(f"  kwargs: {kwargs}")
    return call.object_id


def status(args: argparse.Namespace) -> None:
    if args.call_id:
        import modal
        fc = modal.FunctionCall.from_id(args.call_id)
        try:
            fc.get(timeout=0)
            print("call state: finished")
        except TimeoutError:
            print("call state: running")
        except Exception as e:  # noqa: BLE001
            print(f"call state: {type(e).__name__}: {e}")
    # two timestamped reads of metrics.csv, per the health rule
    for i in range(2):
        out = subprocess.run(
            [sys.executable, "-m", "modal", "volume", "get", "--force", RUNS_VOLUME,
             f"{args.run_name}/metrics.csv", "/tmp/_metrics.csv"],
            capture_output=True, text=True)
        if out.returncode != 0:
            print(f"no metrics.csv yet for {args.run_name}: {out.stderr.strip()[-200:]}")
            return
        lines = Path("/tmp/_metrics.csv").read_text().strip().splitlines()
        print(f"[{time.strftime('%H:%M:%S')}] {len(lines) - 1} rows; last: {lines[-1]}")
        if i == 0:
            time.sleep(args.wait)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("deploy")
    s = sub.add_parser("spawn")
    s.add_argument("--run-name", required=True)
    s.add_argument("--corpus", required=True)
    s.add_argument("--size", default="medium")
    s.add_argument("--max-steps", type=int, default=180000)
    s.add_argument("--batch-size", type=int, default=64)
    s.add_argument("--grad-accum", type=int, default=1)
    s.add_argument("--block-size", type=int, default=2048)
    s.add_argument("--warmup-steps", type=int, default=200)
    s.add_argument("--eval-interval", type=int, default=500)
    s.add_argument("--save-interval", type=int, default=1000)
    s.add_argument("--lr", type=float, default=4e-4)
    s.add_argument("--schedule", default="wsd")
    s.add_argument("--decay-steps", type=int, default=0)
    s.add_argument("--no-augment", action="store_true")
    s.add_argument("--aug-pitch", type=int, default=6)
    s.add_argument("--aug-velocity", type=int, default=1)
    s.add_argument("--doc-start-frac", type=float, default=0.2)
    s.add_argument("--mixture", default="")
    s.add_argument("--compile", action="store_true")
    s.add_argument("--stage-local", action="store_true")
    s.add_argument("--resume", action="store_true")
    s.add_argument("--resume-from", default="")
    s.add_argument("--header-dropout", type=float, default=0.3)
    s.add_argument("--header-drop-all", type=float, default=0.1)
    s.add_argument("--rope-base", type=float, default=0.0)
    st = sub.add_parser("status")
    st.add_argument("--run-name", required=True)
    st.add_argument("--call-id", default="")
    st.add_argument("--wait", type=int, default=90, help="seconds between the two reads")
    args = p.parse_args()
    if args.cmd == "deploy":
        deploy()
    elif args.cmd == "spawn":
        spawn(args)
    else:
        status(args)


if __name__ == "__main__":
    main()
