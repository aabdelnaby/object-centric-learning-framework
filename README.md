# Hierarchical-DINOSAUR and the HierRouter

Code for the master's thesis *Hierarchical Representations via Object-Centric Learning*. It contains what is needed to reproduce the experiments reported in the thesis, and
nothing else.

**The method.** A frozen DINOv3 ViT-S/16 and a frozen DINOSAUR slot-attention module give
object-level slots for an image. *Hierarchical-DINOSAUR* re-runs that same frozen slot attention
*confined to one slot's patches*, which splits each object slot into part-level sub-slots: a
depth-2 tree induced without any object or part supervision. The *HierRouter* then answers
"What is the colour of the `<part>` of the `<object>`?" by routing the question down that tree,
`P(j | object)` over object slots, `P(k | j, part)` within the chosen object, and an attribute head
on the routed part slot, marginalising over all paths. Only the routing head is trained.

**The finding.** Structure does not buy accuracy over flat patch attention: the full router reaches
0.594 on PACO part-colour questions and a single learned query over frozen patches reaches 0.593.
What the hierarchy provides instead is an explicit, inspectable object→part trace behind every
answer. See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Setup

```bash
conda env create -f environment.yml && conda activate oclf_env   # main environment
conda env create -f environment-internvl.yml                      # only for the VLM colour fallback
pip install -e .                                                  # the ocl package + hier_dinosaur
```

Datasets, pretrained weights and where they go: [docs/DATA.md](docs/DATA.md). On a machine that
already has the original research checkout, `bash tools/link_local_data.sh` links everything into
place and `bash tools/collect_thesis_checkpoints.sh` copies the trained heads reported in the thesis
into `checkpoints/thesis/`.

## Running the experiments

The numbered scripts in `experiments/` are the whole pipeline; each submits a SLURM job (set
`LOCAL=1` to run in the foreground, `DRY=1` to print the command). Steps 1-4 build the inputs and
can be skipped if you already have the checkpoint and the feature caches.

| Step | Script | What it does | Cost |
|---|---|---|---|
| 1 | `01_train_dinosaur.sh` | pretrain the DINOSAUR slot module on COCO | ~2 days, 1 GPU |
| 2 | `02_build_paco_dataset.sh` | PACO-LVIS questions, part masks, colour answers | ~1 h, CPU + a short GPU array |
| 3 | `03_build_cub_dataset.sh` | CUB-200 questions | minutes, CPU |
| 4 | `04_precompute_features.sh` | cache the frozen patch features | ~20 min, 1 GPU |
| 5 | `05_train_paco.sh` | the five PACO models of Table 1 | 15 min each (router: longer) |
| 6 | `06_train_cub.sh` | the CUB router | ~1 h, 1 GPU |
| 7 | `07_eval_tables.sh` | Tables 1, 2, 3 and the CUB numbers | ~20 min, 1 GPU |
| 8 | `08_make_figures.sh` | the thesis figures | minutes, CPU |

Each step is also a plain command, for example:

```bash
python -m hier_dinosaur.train --model hier_router --dataset paco --out runs/paco/hier_router
python -m hier_dinosaur.evaluate --checkpoint runs/paco/hier_router/best_model.pt
python -m hier_dinosaur.grounding --router_ckpt ... --patch_ckpt ... --n 800
python -m hier_dinosaur.viz.figures --checkpoint ... --n_samples 12 --colorbar
```

## Where things are

```
hier_dinosaur/       the method and the experiments
  dinosaur.py        frozen DINOv3 + DINOSAUR slot attention (loading, slot extraction)
  hierarchy.py       recursive inference: object slots → part sub-slots (thesis Algorithm 1)
  router.py          TextProjector + HierRouter (routing, path marginalisation, Algorithm 2)
  baselines.py       the flat patch controls (Patch-QDot, Patch-QCA)
  text.py            question parsing + frozen T5 span encoding
  features.py        image preprocessing + the patch-feature cache
  data.py            question CSVs, label vocabularies, cached-feature dataset
  models.py          the five models, run configs, checkpoint I/O
  train.py           training loop
  evaluate.py        Tables 1 and 3, CUB per-part accuracy
  grounding.py       Table 2
  viz/               routing traces, thesis figures, the tree figure
data_prep/           building the two question sets from the raw annotations
experiments/         the numbered pipeline above
ocl/, routed/, configs/   the upstream object-centric-learning-framework, used only for step 1
                          and for loading the pretrained slot module (Apache-2.0, see NOTICE)
tests/               unit tests + the equivalence check against the original thesis code
```

Not in this branch: the exploratory work that did not enter the thesis (ADE20K, Super-CLEVR-3D,
scene out-of-distribution splits, alternative router readouts, VLM baselines, hyper-parameter
search). It remains on the `main` branch of this repository.

## Provenance

This branch is a reorganisation of the research code, not a rewrite: `tests/equivalence_legacy_vs_new.py`
checks it against the original implementation on all six trained checkpoints and they agree
bit-for-bit, from the loaded weights through the routing trace to the answer log-probabilities.

`ocl/`, `routed/`, `configs/` and `scripts/datasets/` are the upstream
[object-centric-learning-framework](https://github.com/amazon-science/object-centric-learning-framework)
(Apache-2.0), pruned to what this thesis uses and extended with DINOv3 support.
