"""Unit tests for the method: question parsing, the router, the tree and checkpoint round-trips.

These use randomly initialised modules and need no data or weights:

    conda run -n oclf_env python -m pytest tests/test_method.py -q

The heavier check that this package reproduces the original thesis implementation on the trained
checkpoints is tests/equivalence_legacy_vs_new.py (needs the data and the legacy checkout).
"""

import math

import pytest
import torch

from hier_dinosaur.baselines import PatchQueryDotProductColorHead, QueryCrossAttentionColorHead
from hier_dinosaur.data import build_label_vocab, split_frame
from hier_dinosaur.hierarchy import confined_slot_attention, select_parents
from hier_dinosaur.models import CONFIG_FORMAT, MODELS, normalize_config
from hier_dinosaur.router import HierRouter, TextProjector
from hier_dinosaur.text import encode_spans, parse_cub_query, parse_paco_query, parse_xy, span_phrases

D_SLOT, D_VIT, C = 256, 384, 12


# ── question parsing ──────────────────────────────────────────────────────────

def test_paco_template():
    assert parse_paco_query("What is the color of the door of the car?") == ("door", "car")
    assert parse_paco_query("What colour is that?") is None


def test_paco_channels():
    part, obj, part_of_obj, readout = span_phrases("What is the color of the door of the car?", "paco")
    assert (part, obj) == ("door", "car")
    assert part_of_obj == "door of the car"     # parent-only routing query
    assert readout == "car door"                # child routing + attribute readout


def test_cub_template_carries_the_attribute():
    assert parse_cub_query("What is the back color of the bird?") == ("back", "bird", "color")
    # the attribute lives on the readout channel so "back color" and "back pattern" differ there
    colour = span_phrases("What is the back color of the bird?", "cub")
    pattern = span_phrases("What is the back pattern of the bird?", "cub")
    assert colour[:3] == pattern[:3] and colour[3] != pattern[3]
    assert colour[3] == "bird back color"


def test_cub_whole_bird_question_has_no_part():
    part, obj, part_of_obj, readout = span_phrases("What is the shape of the bird?", "cub")
    assert part is None and part_of_obj is None      # → zero vectors on those channels
    assert (obj, readout) == ("bird", "bird shape")
    assert parse_xy("What is the shape of the bird?", "cub") == ("", "bird")


# ── router ────────────────────────────────────────────────────────────────────

def make_router(**kw):
    torch.manual_seed(0)
    return HierRouter(d_slot=D_SLOT, num_classes=C, temperature=0.95, child_scorer="mlp", **kw)


def test_router_marginal_is_a_distribution_and_backpropagates():
    router = make_router()
    B, P, K = 4, 9, 5
    parents = torch.randn(B, P, D_SLOT)
    children = torch.randn(B, P, K, D_SLOT)
    q = torch.randn(B, D_SLOT)
    out = router(parents, children, q, q, None, q)
    probs = out.logp_answer.exp()
    assert probs.shape == (B, C)
    assert torch.allclose(probs.sum(1), torch.ones(B), atol=1e-5)
    assert torch.allclose(out.logp_parent.exp().sum(1), torch.ones(B), atol=1e-5)
    assert torch.allclose(out.logp_child.exp().sum(2), torch.ones(B, P), atol=1e-5)
    torch.nn.functional.nll_loss(out.logp_answer, torch.zeros(B, dtype=torch.long)).backward()
    assert router.q_parent.weight.grad is not None and router.q_child.weight.grad is not None


def test_router_marginalisation_matches_the_explicit_sum():
    """log P(a) = log sum_jk P(j|y) P(k|j,x) P(a|c_jk), computed in log space."""
    router = make_router()
    B, P, K = 2, 4, 3
    parents, children = torch.randn(B, P, D_SLOT), torch.randn(B, P, K, D_SLOT)
    q = torch.randn(B, D_SLOT)
    out = router(parents, children, q, q, None, q)
    explicit = (out.logw.exp()[..., None] * out.logp_color.exp()).sum(dim=(1, 2))
    assert torch.allclose(out.logp_answer.exp(), explicit, atol=1e-5)


def test_empty_parents_receive_no_routing_mass():
    router = make_router()
    B, P, K = 3, 6, 4
    nonempty = torch.ones(B, P, dtype=torch.bool)
    nonempty[:, 3:] = False
    out = router(torch.randn(B, P, D_SLOT), torch.randn(B, P, K, D_SLOT),
                 torch.randn(B, D_SLOT), torch.randn(B, D_SLOT), nonempty, torch.randn(B, D_SLOT))
    assert torch.all(out.logp_parent.exp()[:, 3:] == 0)
    assert torch.allclose(out.logp_parent.exp().sum(1), torch.ones(B), atol=1e-5)


def test_a_row_with_no_nonempty_parent_still_normalises():
    router = make_router()
    B, P, K = 2, 5, 3
    nonempty = torch.zeros(B, P, dtype=torch.bool)          # degenerate: keep all parents
    out = router(torch.randn(B, P, D_SLOT), torch.randn(B, P, K, D_SLOT),
                 torch.randn(B, D_SLOT), torch.randn(B, D_SLOT), nonempty, torch.randn(B, D_SLOT))
    assert torch.isfinite(out.logp_answer).all()
    assert torch.allclose(out.logp_answer.exp().sum(1), torch.ones(B), atol=1e-5)


def test_parent_only_router_ignores_the_child_level():
    router = make_router(use_children=False)
    assert not hasattr(router, "q_child")
    B, P = 4, 9
    out = router(torch.randn(B, P, D_SLOT), None, None, torch.randn(B, D_SLOT), None, torch.randn(B, D_SLOT))
    assert out.logp_child is None
    assert torch.allclose(out.logp_answer.exp().sum(1), torch.ones(B), atol=1e-5)


def test_map_path_picks_the_heaviest_path():
    router = make_router()
    B, P, K = 3, 5, 4
    out = router(torch.randn(B, P, D_SLOT), torch.randn(B, P, K, D_SLOT),
                 torch.randn(B, D_SLOT), torch.randn(B, D_SLOT), None, torch.randn(B, D_SLOT))
    logp, j, k = out.map_path_logp()
    flat = out.logw.reshape(B, -1).argmax(1)
    assert torch.equal(j, flat // K) and torch.equal(k, flat % K)
    for b in range(B):
        assert torch.allclose(logp[b], out.logp_color[b, j[b], k[b]])


def test_readout_query_changes_the_answer():
    router = make_router()
    B, P, K = 2, 4, 3
    parents, children = torch.randn(B, P, D_SLOT), torch.randn(B, P, K, D_SLOT)
    q = torch.randn(B, D_SLOT)
    a = router(parents, children, q, q, None, torch.randn(B, D_SLOT)).logp_answer
    b = router(parents, children, q, q, None, torch.randn(B, D_SLOT)).logp_answer
    assert not torch.allclose(a, b)


# ── hierarchy ─────────────────────────────────────────────────────────────────

def test_select_parents_prefers_slots_with_patch_mass():
    slots = torch.randn(1, 4, D_SLOT)
    masks = torch.zeros(1, 4, 196)
    masks[0, 1] = 1.0                       # only slot 1 carries mass
    masks[0, 2] = 0.001
    parents, nonempty, idx = select_parents(slots, masks, n_parents=4)
    assert idx[0, 0] == 1 and bool(nonempty[0, 0])
    assert not bool(nonempty[0, 1:].any())
    assert torch.allclose(parents[0, 0], slots[0, 1])


def test_confinement_keeps_children_inside_the_parent_region():
    """Children may only aggregate patches the parent explains (thesis Section 3.1)."""
    from ocl.neural_networks import build_two_layer_mlp
    from ocl.perceptual_grouping import SlotAttention

    torch.manual_seed(0)
    sa = SlotAttention(dim=D_SLOT, feature_dim=D_SLOT, n_heads=1, iters=3,
                       ff_mlp=build_two_layer_mlp(D_SLOT, D_SLOT, 4 * D_SLOT,
                                                  initial_layer_norm=True, residual=True))
    B, N, K = 2, 49, 4
    embedded = torch.randn(B, N, D_SLOT)
    parent_mask = torch.zeros(B, N)
    parent_mask[:, :10] = 1.0                                  # the parent owns the first ten patches
    slots, attn = confined_slot_attention(sa, embedded, parent_mask, torch.randn(B, K, D_SLOT))
    assert slots.shape == (B, K, D_SLOT) and attn.shape == (B, K, N)
    assert torch.all(attn[:, :, 10:] == 0)                     # nothing outside the parent
    assert torch.all(attn[:, :, :10] >= 0)


# ── baselines ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("project", [False, True])
def test_patch_qdot_head(project):
    head = PatchQueryDotProductColorHead(D_SLOT, D_VIT, D_SLOT, C, project_patches=project)
    logp, attn = head(torch.randn(3, 196, D_VIT), torch.randn(3, D_SLOT))
    assert logp.shape == (3, C) and attn.shape == (3, 196)
    assert torch.allclose(logp.exp().sum(1), torch.ones(3), atol=1e-5)
    assert torch.allclose(attn.sum(1), torch.ones(3), atol=1e-5)
    assert head.d_attn == (D_SLOT if project else D_VIT)


def test_patch_qdot_legacy_offset_drops_leading_tokens():
    """The thesis Patch-QDot runs attended over 192 of 196 patches; see the erratum in docs."""
    head = PatchQueryDotProductColorHead(D_SLOT, D_VIT, D_SLOT, C, legacy_strip_tokens=4)
    _, attn = head(torch.randn(2, 196, D_VIT), torch.randn(2, D_SLOT))
    assert attn.shape == (2, 192)


def test_patch_qca_head():
    head = QueryCrossAttentionColorHead(D_SLOT, C, num_heads=8)
    logp, attn = head(torch.randn(3, 196, D_SLOT), torch.randn(3, D_SLOT))
    assert logp.shape == (3, C) and attn.shape == (3, 196)
    assert torch.allclose(logp.exp().sum(1), torch.ones(3), atol=1e-5)


def test_text_projector_shape():
    out = TextProjector(d_text=768, d_slot=D_SLOT)(torch.randn(5, 768))
    assert out.shape == (5, D_SLOT)


# ── configs ───────────────────────────────────────────────────────────────────

LEGACY_ROUTER = dict(pooler="hier_router", dataset="ade20k", n_slots=9, recursive_children=5,
                     recursive_parents=9, recursive_spread=0.0, router_temp=0.95, child_scorer="mlp",
                     router_readout_query=True, num_heads=64, d_text=768, img_size=224,
                     resize_mode="square", text_encoder="t5", csv_path="x.csv", dino_cache="c.pt")


def test_legacy_router_config_is_translated():
    cfg = normalize_config(LEGACY_ROUTER)
    assert cfg["format"] == CONFIG_FORMAT and cfg["legacy"]
    assert cfg["model"] == "hier_router" and cfg["dataset"] == "paco"
    assert (cfg["n_slots"], cfg["children"], cfg["parents"]) == (9, 5, 9)


def test_legacy_variants_map_to_the_right_models():
    assert normalize_config({**LEGACY_ROUTER, "router_parent_only": True})["model"] == "hier_router_parent_only"
    patch = dict(pooler="patch_qdot", patch_control=True, dataset="ade20k", text_encoder="t5")
    assert normalize_config(patch)["model"] == "patch_qdot_raw"
    assert normalize_config({**patch, "patch_qdot_project_patches": True})["model"] == "patch_qdot_projected"
    assert normalize_config({**patch, "pooler": "qca"})["model"] == "patch_qca"
    assert set(MODELS) == {normalize_config(c)["model"] for c in [
        LEGACY_ROUTER, {**LEGACY_ROUTER, "router_parent_only": True}, patch,
        {**patch, "patch_qdot_project_patches": True}, {**patch, "pooler": "qca"}]}


def test_legacy_patch_qdot_config_restores_the_token_offset():
    cfg = normalize_config(dict(pooler="patch_qdot", patch_control=True, dataset="ade20k", text_encoder="t5"))
    assert cfg["legacy_strip_tokens"] == 4          # so thesis checkpoints reproduce their numbers


def test_unsupported_legacy_configs_are_rejected():
    with pytest.raises(ValueError):
        normalize_config({**LEGACY_ROUTER, "router_color_source": "patch_qdot"})
    with pytest.raises(ValueError):
        normalize_config({**LEGACY_ROUTER, "dataset": "superclevr3d"})
    with pytest.raises(ValueError):
        normalize_config({"pooler": "vqa_paper", "dataset": "ade20k"})


def test_new_config_passes_through():
    cfg = {"format": CONFIG_FORMAT, "model": "hier_router", "dataset": "paco"}
    assert normalize_config(cfg) == cfg


# ── data ──────────────────────────────────────────────────────────────────────

def test_label_vocab_and_split_use_rank_one_training_rows():
    import pandas as pd
    df = pd.DataFrame({
        "image_name": ["a", "b", "c", "d"],
        "query": ["q"] * 4,
        "label": ["red", "blue", "red", "green"],
        "label_rank": [1, 1, 1, 2],          # rank-2 rows are ignored
        "split": ["train", "train", "val", "train"],
    })
    vocab = build_label_vocab(df, "train")
    assert vocab == {"blue": 0, "red": 1}    # sorted, "green" only appears at rank 2
    val = split_frame(df, "val", vocab)
    assert len(val) == 1 and val.iloc[0]["label"] == "red"


# ── figure row selection ──────────────────────────────────────────────────────

def test_select_accepts_a_seed_followed_by_rows():
    from hier_dinosaur.viz.figures import parse_select
    assert parse_select("0:146,10,114") == [(0, 146), (0, 10), (0, 114)]
    assert parse_select("1:4,6;3:1,9") == [(1, 4), (1, 6), (3, 1), (3, 9)]
    assert parse_select("0:12,0:30") == [(0, 12), (0, 30)]        # every row repeating its seed
    with pytest.raises(ValueError):
        parse_select("12,30")                                      # no seed
