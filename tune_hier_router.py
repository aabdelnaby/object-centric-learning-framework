#!/usr/bin/env python3
"""Optuna hyperparameter search for the hier_router VQA head on ADE20K.

Searches the **structural** params the model is most sensitive to — the number of
parent slots (``n_slots``; hier_router refines *all* slots, so this is the parent
count P) and the number of child slots per parent (``recursive_children`` = K) —
together with the routing/optimisation knobs (lr, router temperature, child scorer,
entropy weight, weight decay).

Efficiency: the expensive ViT + T5 feature pass is **n_slots-independent**, so it is
precomputed ONCE to disk (see submit_precompute_ade20k_square.sh) and every trial
loads it with ``--feat_cache --dino_cache/--text_cache``; only the (cheap, frozen-
upstream) slot-attention + router training varies per trial.

Each trial shells out to ``train.py`` with a short epoch budget, writes to its own
dir, and is scored by the best val accuracy in that run's ``metrics.csv``. Per-epoch
val accuracy is streamed back to Optuna so the MedianPruner can kill weak trials early.

Distributed: trials coordinate through a shared Optuna storage (a JournalStorage file
on the shared FS by default — robust for a SLURM array of workers; pass a ``sqlite://``
/ RDB URL to override). Launch many array tasks against the same --storage/--study_name
to parallelise; see submit_tune_hier_router.sh.

    python tune_hier_router.py \
        --dino_cache FG-datset/ade20k/dino_feat_cache_square.pt \
        --text_cache FG-datset/ade20k/text_feat_cache_t5_spans.pt \
        --resize_mode square --n_trials 40

    python tune_hier_router.py --report_only      # just print the best trials so far
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import optuna

REPO = Path(__file__).resolve().parent
DEF_CSV   = "FG-datset/ade20k/parts_color_vqa_internvl.csv"
DEF_DCFG  = "projects/bridging/dinosaur/superclevr3d_feat_rec_dino_small16_dinov3"
DEF_DCKPT = "checkpoints/dinov3/epoch_67-step_500000_dinov3_11_slots.ckpt"
POLL_SECONDS = 20


# ── Search space ─────────────────────────────────────────────────────────────────
def sample_params(trial) -> dict:
    """The hyperparameters Optuna searches. Edit here to widen/narrow the space."""
    return {
        # structural (the point of this study)
        "n_slots":               trial.suggest_categorical("n_slots", [7, 9, 11, 13, 15]),
        "recursive_children":    trial.suggest_int("recursive_children", 2, 8),
        # routing / head
        "router_temp":           trial.suggest_float("router_temp", 0.3, 1.5),
        "child_scorer":          trial.suggest_categorical("child_scorer", ["bilinear", "mlp"]),
        "router_entropy_weight": trial.suggest_categorical("router_entropy_weight",
                                                           [0.0, 1e-3, 1e-2]),
        # optimisation
        "lr":                    trial.suggest_float("lr", 1e-5, 3e-4, log=True),
        "weight_decay":          trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True),
    }


# ── Storage (robust on shared FS) ─────────────────────────────────────────────────
def make_storage(path: str):
    """RDB URL (sqlite:///…, mysql://…) → used directly; a plain path → JournalStorage
    file backend (recommended for distributed access on a network filesystem)."""
    if "://" in path:
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    from optuna.storages import JournalStorage
    try:
        from optuna.storages.journal import JournalFileBackend  # optuna >= 4
        return JournalStorage(JournalFileBackend(path))
    except Exception:
        from optuna.storages import JournalFileStorage          # optuna < 4
        return JournalStorage(JournalFileStorage(path))


# ── metrics.csv helpers ───────────────────────────────────────────────────────────
def read_val_curve(metrics_csv: Path):
    """Return [(epoch, val_acc), …] from a run's metrics.csv (empty if not yet written)."""
    if not metrics_csv.exists():
        return []
    out = []
    try:
        with open(metrics_csv) as fh:
            for r in csv.DictReader(fh):
                out.append((int(r["epoch"]), float(r["val_acc"])))
    except (KeyError, ValueError, OSError):
        pass
    return out


def build_cmd(args, params, trial_dir: Path):
    return [
        sys.executable, str(REPO / "train.py"),
        "--dataset", "ade20k", "--csv_path", args.csv,
        "--text_encoder", "t5", "--pooler", "hier_router", "--rank_method", "attribution",
        "--feat_cache", "--dino_cache", args.dino_cache, "--text_cache", args.text_cache,
        "--dinosaur_cfg", args.dinosaur_cfg, "--dinosaur_ckpt", args.dinosaur_ckpt,
        "--img_size", "224", "--resize_mode", args.resize_mode,
        "--n_slots", str(params["n_slots"]),
        "--recursive_children", str(params["recursive_children"]),
        "--router_temp", f"{params['router_temp']:.5f}",
        "--child_scorer", params["child_scorer"],
        "--router_entropy_weight", f"{params['router_entropy_weight']}",
        "--lr", f"{params['lr']:.6e}", "--weight_decay", f"{params['weight_decay']:.6e}",
        "--optimizer", "adamw", "--lr_schedule", "cosine",
        "--batch_size", str(args.batch_size),
        "--max_epochs", str(args.max_epochs), "--warmup_steps", str(args.warmup_steps),
        "--patience", str(args.patience),
        "--checkpoint_every", "0", "--skip_per_query_eval",
        "--checkpoint_dir", str(trial_dir),
    ]


# ── Objective ─────────────────────────────────────────────────────────────────────
def make_objective(args):
    out_root = Path(args.out_root)

    def objective(trial):
        params      = sample_params(trial)
        trial_dir   = out_root / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        metrics_csv = trial_dir / f"slots_{params['n_slots']}" / "metrics.csv"
        cmd         = build_cmd(args, params, trial_dir)
        (trial_dir / "cmd.txt").write_text(" ".join(cmd) + "\n")
        print(f"\n[trial {trial.number}] {params}", flush=True)

        with open(trial_dir / "train.log", "w") as log:
            proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=log,
                                    stderr=subprocess.STDOUT)
            seen = 0
            try:
                while proc.poll() is None:
                    time.sleep(POLL_SECONDS)
                    curve = read_val_curve(metrics_csv)
                    for epoch, val in curve[seen:]:
                        trial.report(val, epoch)
                        if trial.should_prune():
                            proc.terminate()
                            try:
                                proc.wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                proc.kill()
                            print(f"[trial {trial.number}] pruned @ epoch {epoch} "
                                  f"(val={val:.4f})", flush=True)
                            raise optuna.TrialPruned()
                    seen = len(curve)
            finally:
                if proc.poll() is None:
                    proc.terminate()

        curve = read_val_curve(metrics_csv)
        if not curve:
            print(f"[trial {trial.number}] no metrics (rc={proc.returncode}); "
                  f"see {trial_dir/'train.log'} — scoring 0.0", flush=True)
            return 0.0
        best = max(v for _, v in curve)
        print(f"[trial {trial.number}] done: best val_acc={best:.4f} "
              f"over {len(curve)} epochs", flush=True)
        return best

    return objective


def dump_best(study, out_root: Path):
    try:
        bt = study.best_trial
    except ValueError:
        return
    payload = {
        "best_value": bt.value, "best_trial": bt.number, "best_params": bt.params,
        "n_complete": len([t for t in study.trials
                           if t.state == optuna.trial.TrialState.COMPLETE]),
        "n_total": len(study.trials),
    }
    (out_root / "best_params.json").write_text(json.dumps(payload, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study_name", default="hier_router_square")
    ap.add_argument("--storage", default="runs/optuna/hier_router_square/journal.log",
                    help="Journal file path (default) or an RDB URL (sqlite:///…).")
    ap.add_argument("--out_root", default="runs/optuna/hier_router_square")
    ap.add_argument("--n_trials", type=int, default=40, help="trials for THIS worker")
    ap.add_argument("--timeout", type=int, default=None,
                    help="stop this worker after N seconds (set below the SLURM wall time)")
    ap.add_argument("--dino_cache", default="FG-datset/ade20k/dino_feat_cache_square.pt")
    ap.add_argument("--text_cache", default="FG-datset/ade20k/text_feat_cache_t5_spans.pt")
    ap.add_argument("--resize_mode", default="square", choices=["crop", "square", "pad"])
    ap.add_argument("--csv", default=DEF_CSV)
    ap.add_argument("--dinosaur_cfg", default=DEF_DCFG)
    ap.add_argument("--dinosaur_ckpt", default=DEF_DCKPT)
    ap.add_argument("--max_epochs", type=int, default=60, help="per-trial epoch budget")
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report_only", action="store_true",
                    help="print the study's best/top trials and exit (run no trials)")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.INFO)
    storage = make_storage(args.storage)

    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True, group=True)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=15)
    study   = optuna.create_study(
        study_name=args.study_name, storage=storage, direction="maximize",
        sampler=sampler, pruner=pruner, load_if_exists=True,
    )

    if args.report_only:
        n_done = len([t for t in study.trials
                      if t.state == optuna.trial.TrialState.COMPLETE])
        print(f"study '{args.study_name}': {len(study.trials)} trials ({n_done} complete)")
        top = sorted([t for t in study.trials if t.value is not None],
                     key=lambda t: t.value, reverse=True)[:10]
        for t in top:
            print(f"  #{t.number:4d}  val_acc={t.value:.4f}  {t.params}")
        dump_best(study, out_root)
        return

    for missing in (args.dino_cache, args.text_cache):
        if not Path(missing).exists():
            raise SystemExit(f"feature cache not found: {missing}\n"
                             f"Run submit_precompute_ade20k_square.sh first.")

    study.optimize(
        make_objective(args), n_trials=args.n_trials, timeout=args.timeout,
        catch=(Exception,), callbacks=[lambda st, tr: dump_best(st, out_root)],
    )
    dump_best(study, out_root)
    print(f"\nWorker done. Best so far → {out_root/'best_params.json'}")


if __name__ == "__main__":
    main()
