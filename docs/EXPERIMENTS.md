# Experiments: every thesis number and how to reproduce it

Each table and figure below names the command that produces it and the value reported in the
thesis. `CKPT=checkpoints/thesis` are the trained heads of the thesis runs
(`tools/collect_thesis_checkpoints.sh`); replace it with `runs/` to evaluate your own training.

Everything can be produced in one go with

```bash
bash experiments/07_eval_tables.sh      # Tables 1-3 and the CUB numbers → results/
bash experiments/08_make_figures.sh     # the figures → figures/
```

## Models

| Name | What it is | Trained parameters |
|---|---|---|
| `hier_router` | the full router: object→part routing, attribute read from the part slot | 792,077 |
| `hier_router_parent_only` | ablation without the part level, colour read off the object slot | 529,164 |
| `patch_qdot_raw` | flat control: one query·patch dot product in the 384-d ViT space | 760,460 |
| `patch_qdot_projected` | the same after one learned 384→256 projection | 627,724 |
| `patch_qca` | flat control with a learned multi-head cross-attention layer | 891,404 |

All share the frozen DINOv3 ViT-S/16, the frozen DINOSAUR slot module, the frozen T5-base text
encoder, the same query-conditioned attribute head and the same recipe (AdamW 2e-4, weight decay
0.02, 10k warm-up then cosine, batch 128, NLL). Router settings: 9 object slots, 5 part sub-slots
each, temperature 0.95, MLP child scorer, no spatial prior. All models overfit within the budget,
so the best-validation checkpoint is reported.

## Table 1 — colour accuracy on the PACO validation split (n = 2,656)

```bash
python -m hier_dinosaur.evaluate --checkpoint $CKPT/paco_hier_router/best_model.pt --out results/paco_hier_router.json
# ... the same for the other four models, then:
python -m hier_dinosaur.evaluate --summarize results/
```

| Model | top-1 | top-2 | top-3 |
|---|:-:|:-:|:-:|
| HierRouter (full, object→part) | **0.594** | 0.773 | 0.873 |
| Patch-QDot (projected) | 0.593 | 0.769 | 0.867 |
| Patch-QDot (raw) | 0.578 | 0.761 | 0.865 |
| Patch-QCA | 0.573 | 0.770 | 0.862 |
| HierRouter (parent-only) | 0.561 | 0.750 | 0.843 |

The top-1 column is the best-validation accuracy stored in each checkpoint; top-2 and top-3 come
from one re-evaluation pass. Adding the part level lifts the router from 0.561 to 0.594, but the
flat Patch-QDot control reaches the same 0.593: the hierarchy recovers detail that object-slot
pooling discards without exceeding what the raw patches already carried.

**Reproducibility.** The patch models are deterministic and reproduce their numbers exactly. The
routers seed their part sub-slots from the conditioning prior, so their accuracy moves by about
±0.005 between passes (the same checkpoint gave 0.5945, 0.5926 and 0.5911 on different passes).
`--seed` fixes a pass; the thesis numbers were produced unseeded.

## Table 2 — grounding faithfulness (PACO validation subset, n = 800)

```bash
python -m hier_dinosaur.grounding \
  --router_ckpt $CKPT/paco_hier_router/best_model.pt \
  --patch_ckpt  $CKPT/paco_patch_qdot_projected/best_model.pt \
  --patch_ckpt  $CKPT/paco_patch_qdot_raw/best_model.pt \
  --n 800 --seed 0 --legacy_grid --out results/paco_grounding.json
```

Chance mass (mean part area) ≈ 0.031.

| Model | mass-in-mask | pointing | IoU@mean | colour acc. (subset) |
|---|:-:|:-:|:-:|:-:|
| HierRouter (path-weighted) | **0.165** | **0.203** | **0.121** | 0.505 |
| HierRouter (argmax path) | **0.180** | **0.208** | **0.124** | 0.505 |
| Patch-QDot (raw) | 0.075 | 0.078 | 0.048 | 0.526 |
| Patch-QDot (projected) | 0.074 | 0.076 | 0.048 | 0.531 |

The router attends to the named part 2.2-2.6× more than either patch baseline and more than five
times above chance, while scoring slightly *lower* colour accuracy on the same subset.

`--legacy_grid` selects the attention-to-image mapping these numbers were produced with; see the
note in `experiments/07_eval_tables.sh` before comparing against runs made without it.

## Table 3 — marginal vs MAP-path readout (full router, PACO validation)

Produced by the same evaluation pass as Table 1 (the `map_path` block of the router's result JSON).

| Readout | top-1 | top-2 | top-3 |
|---|:-:|:-:|:-:|
| Marginal P(a) | 0.591 | 0.768 | 0.870 |
| MAP path P(a \| c_j*k*) | 0.574 | 0.752 | 0.855 |

Committing to one object→part path costs about 1.7 points and buys a single, fully grounded
localisation; consistent with Table 2, the argmax-path mask is the better-grounded one.

## CUB-200 (thesis Section 4.5)

```bash
python -m hier_dinosaur.evaluate --checkpoint $CKPT/cub_hier_router/best_model.pt --per_query \
  --out results/cub_hier_router.json
# the per-part table of the thesis is the epoch-30 block recorded during training:
python -m hier_dinosaur.evaluate --from_stats $CKPT/cub_hier_router/per_query_stats.csv --epoch 30
```

Overall colour accuracy **56.7 %** (best test checkpoint, epoch 44; 15 colour classes, chance
6.7 %), above all six zero-shot vision-language models of the FG-BMK benchmark, the strongest of
which reaches 47.4 %. That comparison is contextual, not controlled: the VLMs are zero-shot with a
permissive matching criterion, the router is supervised on the CUB training split.

Per-part accuracies come from the near-peak epoch-30 checkpoint (macro average 56.0 %):

| Part | acc. | Part | acc. |
|---|:-:|---|:-:|
| back | 57.2 | primary | 56.5 |
| belly | 57.6 | throat | 55.7 |
| bill | 45.6 | under tail | 51.1 |
| breast | 55.4 | underparts | 55.9 |
| crown | 58.1 | upper tail | 52.8 |
| eye | 86.1 | upperparts | 55.7 |
| forehead | 56.8 | wing | 56.2 |
| leg | 39.9 | **average** | **56.0** |
| nape | 55.5 | | |

Both the router and the strongest VLM peak on the eye, the most localised and saturated part, and
struggle on leg and bill. The router is far more robust on the broad plumage regions, where the
zero-shot model has no mechanism to bind a colour to a named part.

## Figures

```bash
bash experiments/08_make_figures.sh
```

| Figure | Command | Notes |
|---|---|---|
| Hierarchical-DINOSAUR tree | `python -m hier_dinosaur.viz.tree --image <coco image>` | input → object slots → part sub-slots, from the frozen recursive inference |
| routing trace | `python -m hier_dinosaur.viz.routing --checkpoint <router> --n_samples 4` | the ten-panel traversal: masks, both routing distributions, path weights, both readouts |
| PACO localisation | `python -m hier_dinosaur.viz.figures --checkpoint <router> --select ... --colorbar` | rows chosen by hand from a candidate pool |
| CUB localisation | the same with the CUB checkpoint | |

The localisation figures are hand-picked: generate a pool (`--n_samples 200 --page_rows 20`), read
the contact sheets, then pass the row indices to `--select` (`"0:146,10,114"`, or `"1:4,6;3:1,9"` to mix seeds). The thesis rows only come back with
the thesis checkpoints and the same seeds; with your own runs, pick new ones.

The tree figure in the thesis was rendered from an older mask dump produced by an earlier,
*trained* hierarchical model that predates the frozen recursive inference the method chapter
describes. `hier_dinosaur.viz.tree` regenerates the figure with the method as published, so it
will not be pixel-identical to the printed one.

## Verifying the refactor

`tests/equivalence_legacy_vs_new.py` checks this package against the original thesis code: for each
of the six checkpoints it compares loaded weights, dataset rows, spans, features, log-probabilities
and the full routing trace under an identical seed. All six agree bit-for-bit.

```bash
OLD_REPO=../object-centric-learning-framework python tests/equivalence_legacy_vs_new.py
```
