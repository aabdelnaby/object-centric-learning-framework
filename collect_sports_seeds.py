#!/usr/bin/env python
"""Aggregate the sports-only multi-seed sweep: mean +/- std OOD acc per model + paired tests.

Reads runs/ood_sports_seeds/<model>/seed<k>/{patches,slots_*}/ood_*.json and reports, per
model, the held-out-sports top-1 across seeds (mean, std), then paired comparisons of the
router vs each baseline (mean delta, std, win count, paired t-test) — to judge whether the
router's sports edge survives seed noise.
"""
import glob
import json
import os
import statistics as st

MODELS = ["raw", "proj", "rtr"]
ROOT = "runs/ood_sports_seeds"


def load(model):
    """seed -> (in_domain, ood_top1, ood_top3)."""
    out = {}
    for p in glob.glob(os.path.join(ROOT, model, "seed*", "**", "ood_*.json"), recursive=True):
        seed = int(os.path.basename(os.path.dirname(os.path.dirname(p))).replace("seed", "")) \
            if "seed" in p else None
        # seed dir is .../seed<k>/<patches|slots_N>/ood.json -> 2 levels up
        parts = os.path.normpath(p).split(os.sep)
        seed = int([x for x in parts if x.startswith("seed")][0].replace("seed", ""))
        d = json.load(open(p))
        out[seed] = (d["in_domain_val_acc"], d["ood_top1"], d["ood_top3"])
    return out


def main():
    data = {m: load(m) for m in MODELS}
    print(f"{'model':5s} | {'n':>2s} | {'in-dom mean':>11s} | {'OOD mean+/-std':>16s} | "
          f"{'OOD top-3':>9s} | seeds(OOD)")
    ood = {}
    for m in MODELS:
        d = data[m]
        seeds = sorted(d)
        o = [d[s][1] for s in seeds]
        ood[m] = {s: d[s][1] for s in seeds}
        if not o:
            print(f"{m:5s} |  0 | (no results yet)")
            continue
        idv = [d[s][0] for s in seeds]
        t3 = [d[s][2] for s in seeds]
        sd = st.pstdev(o) if len(o) > 1 else 0.0
        print(f"{m:5s} | {len(o):2d} | {st.mean(idv):11.4f} | {st.mean(o):.4f}+/-{sd:.4f} | "
              f"{st.mean(t3):9.4f} | " + " ".join(f"{s}:{d[s][1]:.3f}" for s in seeds))

    # paired router-vs-baseline (only seeds present for both)
    try:
        from scipy.stats import ttest_rel
        have_scipy = True
    except Exception:
        have_scipy = False
    print("\nPaired router vs baseline (shared seeds):")
    for b in ["raw", "proj"]:
        shared = sorted(set(ood.get("rtr", {})) & set(ood.get(b, {})))
        if not shared:
            print(f"  rtr vs {b}: no shared seeds yet"); continue
        diffs = [ood["rtr"][s] - ood[b][s] for s in shared]
        md = st.mean(diffs)
        sd = st.pstdev(diffs) if len(diffs) > 1 else 0.0
        wins = sum(x > 0 for x in diffs)
        line = (f"  rtr vs {b}: n={len(shared)}  mean Δ={md:+.4f} (±{sd:.4f})  "
                f"wins={wins}/{len(shared)}")
        if have_scipy and len(shared) > 1:
            t, p = ttest_rel([ood["rtr"][s] for s in shared], [ood[b][s] for s in shared])
            line += f"  paired t p={p:.3f}"
        print(line)


if __name__ == "__main__":
    main()
