#!/usr/bin/env python
"""Collect leave-one-scene-out OOD results for all sweep models into one table.

Scans each model's runs/<root>/<scene>/<jobid>/{patches,slots_N}/ood_*.json (written by
eval_ood.py) and prints, per held-out scene, every model's in-domain val acc and its OOD
(held-out scene) acc. Then summarises how often the hier_router's OOD acc beats each flat
baseline. Models (label -> runs root) default to the three sweeps; override with --models.
"""
import argparse
import csv as _csv
import glob
import json
import os

DEFAULT_MODELS = [
    ("raw",  "runs/ood_sweep_baseline"),            # Patch-QDot RAW
    ("proj", "runs/ood_sweep_baseline_projected"),  # Patch-QDot PROJECTED
    ("rtr",  "runs/ood_sweep_hier_router"),         # hier_router
]


def find_jsons(root):
    """scene -> json dict; latest by mtime wins if evaluated more than once."""
    out = {}
    n = len(os.path.normpath(root).split(os.sep))
    for path in sorted(glob.glob(os.path.join(root, "**", "ood_*.json"), recursive=True),
                       key=os.path.getmtime):
        parts = os.path.normpath(path).split(os.sep)
        scene = parts[n] if len(parts) > n else os.path.basename(os.path.dirname(path))
        try:
            out[scene] = json.load(open(path))
        except Exception:
            pass
    return out


def f(x):
    return f"{x:.4f}" if isinstance(x, float) else "  -  "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None,
                    help="label:root pairs (default: raw, proj, rtr)")
    ap.add_argument("--ref", default="rtr",
                    help="model label whose OOD acc is compared against the others")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="scene labels to drop from the table and the aggregates")
    ap.add_argument("--out_csv", default="FG-datset/ood_sweep_results.csv")
    args = ap.parse_args()

    models = DEFAULT_MODELS if not args.models else \
        [(m.split(":", 1)[0], m.split(":", 1)[1]) for m in args.models]
    data = {label: find_jsons(root) for label, root in models}
    labels = [l for l, _ in models]
    excl = set(args.exclude)
    scenes = sorted({s for d in data.values() for s in d} - excl)
    if excl:
        print(f"(excluding {len(excl)} scenes: {sorted(excl)})")

    # header
    cols = " ".join(f"{l+'ID':>8s} {l+'OOD':>8s}" for l in labels)
    deltas = [l for l in labels if l != args.ref]
    dcols = " ".join(f"{args.ref+'-'+l:>9s}" for l in deltas)
    hdr = f"{'scene':13s} {'rows':>5s} | {cols} | {dcols}"
    print(hdr); print("-" * len(hdr))

    rows_out = []
    wins = {l: 0 for l in deltas}
    n_done = {l: 0 for l in deltas}
    for s in scenes:
        rec = {"scene": s}
        rows_n = ""
        cells = []
        for l in labels:
            d = data[l].get(s)
            idv = d.get("in_domain_val_acc") if d else None
            ood = d.get("ood_top1") if d else None
            rows_n = (d or {}).get("rows", rows_n)
            rec[f"{l}_indomain"] = idv
            rec[f"{l}_ood"] = ood
            cells.append(f"{f(idv):>8s} {f(ood):>8s}")
        dcells = []
        ref_ood = rec.get(f"{args.ref}_ood")
        for l in deltas:
            base_ood = rec.get(f"{l}_ood")
            dl = (ref_ood - base_ood) if (ref_ood is not None and base_ood is not None) else None
            rec[f"delta_{args.ref}_vs_{l}"] = dl
            if dl is not None:
                n_done[l] += 1
                wins[l] += int(dl > 0)
            dcells.append(f"{f(dl):>9s}")
        rec["rows"] = rows_n
        print(f"{s:13s} {str(rows_n):>5s} | {' '.join(cells)} | {' '.join(dcells)}")
        rows_out.append(rec)

    print("-" * len(hdr))
    for l in deltas:
        if n_done[l]:
            md = sum(r[f"delta_{args.ref}_vs_{l}"] for r in rows_out
                     if r.get(f"delta_{args.ref}_vs_{l}") is not None) / n_done[l]
            print(f"{args.ref} OOD beats {l} on {wins[l]}/{n_done[l]} scenes "
                  f"| mean OOD Δ({args.ref}-{l}) = {md:+.4f}")

    if rows_out:
        keys = ["scene", "rows"] + [k for k in rows_out[0] if k not in ("scene", "rows")]
        with open(args.out_csv, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows_out:
                w.writerow(r)
        print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
