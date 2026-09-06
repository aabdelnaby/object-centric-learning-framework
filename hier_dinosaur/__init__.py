"""Hierarchical-DINOSAUR and the HierRouter: code for the thesis experiments.

Modules
-------
dinosaur    frozen DINOv3 ViT-S/16 + DINOSAUR slot attention (loading, slot extraction)
hierarchy   recursive inference: refine every object slot into K part sub-slots (Algorithm 1)
router      TextProjector + HierRouter (object→part routing, path marginalisation)
baselines   flat patch controls: Patch-QDot and Patch-QCA heads
text        T5 span encoding of the question ("<part>", "<object>", ...)
features    image preprocessing + cached DINOv3 patch features
data        cached-feature datasets (PACO-LVIS, CUB-200) and label vocabularies
models      SlotRouterModel / PatchBaselineModel, config handling, checkpoint I/O
train       training loop (python -m hier_dinosaur.train)
evaluate    Table 1 / Table 3 / CUB per-part (python -m hier_dinosaur.evaluate)
grounding   Table 2 grounding faithfulness (python -m hier_dinosaur.grounding)
viz         routing-trace panels, thesis figures, Hierarchical-DINOSAUR tree figure
"""

__version__ = "1.0.0"
