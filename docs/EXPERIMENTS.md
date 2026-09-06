# Experiments: every thesis number and how to reproduce it

Each table and figure below names the command that produces it and the value reported in the
thesis. `CKPT=checkpoints/thesis` are the trained heads of the thesis runs
(`tools/collect_thesis_checkpoints.sh`); replace it with `runs/` to evaluate your own training.

> **Read the [erratum](#erratum-the-patch-qdot-token-offset) before citing Table 2.** A bug in the
> grounding evaluation made the Patch-QDot baselines look far worse at localisation than they are.
> Corrected, they ground better than the router and the accuracy-grounding dissociation reported in
> the thesis does not hold. Tables 1 and 3 and the CUB results are unaffected.

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
  --n 800 --seed 0 --out results/paco_grounding.json
```

Chance mass (mean part area) ≈ 0.031.

**This table does not survive the erratum below.** As published it reads:

| Model | mass-in-mask | pointing | IoU@mean | colour acc. (subset) |
|---|:-:|:-:|:-:|:-:|
| HierRouter (path-weighted) | **0.165** | **0.203** | **0.121** | 0.505 |
| HierRouter (argmax path) | **0.180** | **0.208** | **0.124** | 0.505 |
| Patch-QDot (raw) | 0.075 | 0.078 | 0.048 | 0.526 |
| Patch-QDot (projected) | 0.074 | 0.076 | 0.048 | 0.531 |

supporting the claim that the router attends to the named part 2.2-2.6× more than either patch
baseline while scoring the same, an accuracy-grounding dissociation. The two Patch-QDot rows were
produced by a broken attention-to-image mapping. Corrected, those models ground *better* than the
router and the dissociation does not hold. Add `--legacy_grid` to reproduce the published rows;
see [the erratum](#erratum-the-patch-qdot-token-offset) for the corrected table and what it means.

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

## Erratum: the Patch-QDot token offset

Found while reorganising this code. It has two parts, and the second one overturns Table 2.

**What went wrong.** The Patch-QDot head dropped four leading tokens on the assumption that the
feature cache still contained DINOv3's register tokens. It does not: the feature extractor already
removes the class and register tokens, so the cache holds exactly the 196 spatial patches
(`ocl/feature_extractors/timm.py` drops both). Consequences:

1. *In training*: the two Patch-QDot models attended over patches 4-195, so the first four patches
   of the image (the top-left corner) were invisible to them. A small handicap; they still reached
   0.578 and 0.593.
2. *In the grounding evaluation*: those 192 values were then laid out on a 13×13 grid (the largest
   square that fits), using only the first 169 of them. The attention maps were therefore scrambled
   and truncated, which is what produced the near-chance mass-in-mask of 0.074.

The router and Patch-QCA are unaffected: neither strips tokens, and both always produce 196 values.

**The corrected Table 2.** Same checkpoints, same 800 questions, same seed; the only change is that
each attention value is placed on the patch it actually came from:

| Model | mass-in-mask | pointing | IoU@mean | colour acc. |
|---|:-:|:-:|:-:|:-:|
| HierRouter (path-weighted) | 0.167 | 0.229 | 0.123 | 0.496 |
| HierRouter (argmax path) | 0.187 | 0.234 | 0.128 | 0.496 |
| **Patch-QDot (projected)** | **0.261** | **0.338** | **0.163** | 0.531 |
| Patch-QDot (raw) | 0.229 | 0.293 | 0.146 | 0.526 |
| chance (part area) | 0.031 | 0.031 | — | — |

The controlled comparison is unambiguous: running the same evaluation with `--legacy_grid`
reproduces the published Patch-QDot rows to four decimals (0.0737 / 0.0763 / 0.0482 projected,
0.0754 / 0.0775 / 0.0482 raw) and leaves the router rows bit-identical, so the grid mapping is the
entire difference.

**What this means for the thesis.** The claim that the router grounds its answers 2.2-2.6× better
than the flat patch baselines does not hold. With the correct mapping the ranking reverses: the
Patch-QDot models put *more* attention mass inside the queried part (0.26 and 0.23) than the router
(0.17), and they also read the colour slightly more accurately. The accuracy-grounding dissociation
in Section 4.4 and the discussion built on it therefore need revisiting. What survives is narrower:
the router grounds well above chance (5.4× the part-area floor) and it is the only model that
exposes an explicit, inspectable object→part trace, but that trace is not better localised than
what a single learned query over frozen patches already achieves.

The router numbers here come from a seeded pass and differ slightly from the published ones
(pointing 0.229 vs 0.203) because the part sub-slots are sampled; mass-in-mask and IoU agree to
within 0.002.

**The fix.** New runs use all 196 patches (`legacy_strip_tokens = 0`) and the grounding evaluation
always maps attention back to the true 14×14 grid. Thesis checkpoints keep loading and keep
reproducing their published accuracies, because `normalize_config` restores the old strip for them.

**Retrained with the fix.** `bash experiments/05_train_paco.sh patch_qdot_raw patch_qdot_projected`,
then evaluate with `CKPT_DIR=$PWD/runs bash experiments/07_eval_tables.sh`. Seeing all 196 patches
helps both models, and the projected one then overtakes the router:

| Model | thesis top-1 | retrained top-1 | top-2 | top-3 |
|---|:-:|:-:|:-:|:-:|
| Patch-QDot (raw) | 0.578 | 0.587 | 0.751 | 0.838 |
| Patch-QDot (projected) | 0.593 | **0.606** | 0.785 | 0.874 |
| HierRouter (full), for comparison | 0.594 | — | — | — |

Their grounding, retrained and correctly evaluated, is unchanged in kind: mass-in-mask 0.258
(projected) and 0.242 (raw) against the router's 0.167. So the reversal is not an artefact of the
handicapped checkpoints; it holds for models trained with the fix as well.

Both peaked at epoch 16-17 of a 200-epoch run and then overfit, the same dynamics the thesis
describes for every model.

## Verifying the refactor

`tests/equivalence_legacy_vs_new.py` checks this package against the original thesis code: for each
of the six checkpoints it compares loaded weights, dataset rows, spans, features, log-probabilities
and the full routing trace under an identical seed. All six agree bit-for-bit.

```bash
OLD_REPO=../object-centric-learning-framework python tests/equivalence_legacy_vs_new.py
```
