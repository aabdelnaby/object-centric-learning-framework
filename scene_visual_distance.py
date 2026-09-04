#!/usr/bin/env python
"""Visual OOD-ness of each held-out scene, in the model's own DINO feature space.

The classifiers consume frozen DINOv3 patch features (the combined_square cache). So the
honest measure of how "far" a held-out scene is, is the distance between that scene's image
features and the REST of the data IN THAT SAME SPACE. We summarise each image by mean-pooling
its 196 patch tokens -> a 384-d descriptor, then for each held-out scene S (vs all other
scenes) compute three complementary distribution-distance measures:

  FID   Frechet distance ||mu_S-mu_R||^2 + Tr(C_S+C_R-2(C_S C_R)^1/2)  (standard domain gap)
  AUC   how separable S is from the rest with a linear probe (5-fold ROC-AUC; .5=identical,
        1=perfectly distinct). Rest subsampled to |S| for balance.
  kNN   mean cosine distance from each S image to its 5 nearest REST images (isolation)

Finally we correlate each scene's visual distance with its measured OOD accuracy drop
(mean over the 3 sweep models) — a positive Spearman rho is the evidence that visual distance
drives the OOD difficulty.
"""
import csv
import glob
import json
import os

import numpy as np
import torch

CACHE = "FG-datset/dino_feat_cache_combined_square.pt"
CSV = "FG-datset/paco_questions_with_scene.csv"
EXCLUDE = {"other", "unknown"}
MODEL_ROOTS = {"raw": "runs/ood_sweep_baseline",
               "proj": "runs/ood_sweep_baseline_projected",
               "rtr": "runs/ood_sweep_hier_router"}


def scene_of_image():
    m = {}
    for r in csv.DictReader(open(CSV)):
        m.setdefault(r["image_name"], r["scene"])
    return m


def ood_drops():
    """scene -> mean (in_domain - ood_top1) over the 3 models."""
    per = {}
    for root in MODEL_ROOTS.values():
        n = len(os.path.normpath(root).split(os.sep))
        for p in glob.glob(os.path.join(root, "**", "ood_*.json"), recursive=True):
            s = os.path.normpath(p).split(os.sep)[n]
            d = json.load(open(p))
            per.setdefault(s, []).append(d["in_domain_val_acc"] - d["ood_top1"])
    return {s: float(np.mean(v)) for s, v in per.items()}


def fid(Xa, Xb):
    from scipy.linalg import sqrtm
    mu_a, mu_b = Xa.mean(0), Xb.mean(0)
    Ca, Cb = np.cov(Xa, rowvar=False), np.cov(Xb, rowvar=False)
    cov_sqrt = sqrtm(Ca @ Cb)
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(Ca + Cb - 2 * cov_sqrt))


def domain_auc(Xs, Xr, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xr), size=min(len(Xs), len(Xr)), replace=False)
    X = np.concatenate([Xs, Xr[idx]])
    y = np.concatenate([np.ones(len(Xs)), np.zeros(len(idx))])
    clf = LogisticRegression(max_iter=1000, C=1.0)
    return float(cross_val_score(clf, X, y, cv=5, scoring="roc_auc").mean())


def mean_knn_cos(Xs_n, Xr_n, k=5, batch=256):
    """mean cosine distance (1 - sim) from each S image to its k nearest REST images."""
    out = []
    for i in range(0, len(Xs_n), batch):
        sims = Xs_n[i:i + batch] @ Xr_n.T              # cosine sims
        topk = np.sort(sims, axis=1)[:, -k:]           # k nearest
        out.append((1 - topk.mean(1)))
    return float(np.concatenate(out).mean())


def main():
    print("Loading DINO cache + mean-pooling per image …", flush=True)
    cache = torch.load(CACHE, map_location="cpu")["features"]
    s_of = scene_of_image()
    names = [n for n in s_of if n in cache and s_of[n] not in EXCLUDE]
    X = np.stack([cache[n].float().mean(0).numpy() for n in names]).astype(np.float64)
    scenes = np.array([s_of[n] for n in names])
    del cache
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    print(f"  {len(names)} images, dim={X.shape[1]}", flush=True)

    drops = ood_drops()
    uscenes = sorted(set(scenes))
    rows = []
    for S in uscenes:
        m = scenes == S
        Xs, Xr = X[m], X[~m]
        Xsn, Xrn = Xn[m], Xn[~m]
        rows.append((S, int(m.sum()),
                     fid(Xs, Xr),
                     domain_auc(Xs, Xr),
                     mean_knn_cos(Xsn, Xrn),
                     drops.get(S, float("nan"))))
        print(f"  done {S}", flush=True)

    rows.sort(key=lambda r: -r[2])  # by FID desc
    print(f"\n{'scene':13s} {'imgs':>5s} {'FID':>8s} {'domAUC':>7s} {'kNNcos':>7s} {'oodDrop':>8s}")
    for S, n, fd, auc, knn, dr in rows:
        print(f"{S:13s} {n:5d} {fd:8.2f} {auc:7.3f} {knn:7.4f} {dr:8.4f}")

    # correlation of each visual-distance metric with the OOD drop
    from scipy.stats import spearmanr
    arr = np.array([(r[2], r[3], r[4], r[5]) for r in rows if not np.isnan(r[5])])
    print("\nSpearman rho (visual distance vs OOD drop), across scenes:")
    for j, lab in enumerate(["FID", "domAUC", "kNNcos"]):
        rho, p = spearmanr(arr[:, j], arr[:, 3])
        print(f"  {lab:7s}: rho={rho:+.3f}  p={p:.3f}")

    rank = {r[0]: i + 1 for i, r in enumerate(rows)}            # FID rank (1=farthest)
    print(f"\nsports FID rank: {rank.get('sports')}/{len(rows)}")


if __name__ == "__main__":
    main()
