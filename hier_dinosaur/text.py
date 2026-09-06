"""Question parsing and frozen T5-base span encoding.

Every question is parsed into four phrases, each embedded by mean-pooling the T5 encoder
states of the phrase (padding tokens excluded):

    channel 0  part          "<part>"                       (unused by the final models)
    channel 1  object        "<object>"                     → parent routing P(j | y)
    channel 2  part-of-obj   "<part> of the <object>"       → parent-only ablation query
    channel 3  readout       "<object> <part>"              → child routing + attribute readout

Templates:
    paco   "What is the color of the <part> of the <object>?"
    cub    "What is the <part> <attribute> of the bird?"   (object = bird; channel 3 =
           "bird <part> <attribute>" so the same body part can be asked about different attributes)
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Sequence, Tuple

import torch

T5_MODEL = "t5-base"
MAX_TEXT_LEN = 64
N_CHANNELS = 4
DATASETS = ("paco", "cub")

_PACO_RE = re.compile(r"color of the (.+?) of the (.+?)\s*\?*\s*$", re.IGNORECASE)
_CUB_RE = re.compile(r"what is the (.+?) of the bird\s*\?*\s*$", re.IGNORECASE)
_CUB_ATTRS = {"color", "pattern", "length", "shape", "size"}


def parse_paco_query(query: str) -> Optional[Tuple[str, str]]:
    """'What is the color of the <part> of the <object>?' → (part, object), else None."""
    m = _PACO_RE.search(str(query).strip())
    return (m.group(1).strip(), m.group(2).strip()) if m else None


def parse_cub_query(query: str) -> Optional[Tuple[str, str, str]]:
    """'What is the <part> <attribute> of the bird?' → (part, "bird", attribute), else None.

    Whole-bird questions ("What is the shape of the bird?") return ``part=""``.
    """
    m = _CUB_RE.search(str(query).strip())
    if not m:
        return None
    toks = m.group(1).strip().split()
    if len(toks) > 1 and toks[-1].lower() in _CUB_ATTRS:
        return " ".join(toks[:-1]), "bird", toks[-1].lower()
    return "", "bird", m.group(1).strip()


def span_phrases(query: str, dataset: str):
    """The four channel phrases for a query (``None`` entries → zero vectors), or ``None`` if unmatched."""
    if dataset == "cub":
        parsed = parse_cub_query(query)
        if parsed is None:
            return None
        part, obj, attr = parsed
        if part:
            return (part, obj, f"{part} of the {obj}", f"{obj} {part} {attr}")
        return (None, obj, None, f"{obj} {attr}")
    parsed = parse_paco_query(query)
    if parsed is None:
        return None
    part, obj = parsed
    return (part, obj, f"{part} of the {obj}", f"{obj} {part}")


def parse_xy(query: str, dataset: str) -> Optional[Tuple[str, str]]:
    """Display phrases (part, object) for a query, or ``None`` if it does not match the template."""
    ph = span_phrases(query, dataset)
    if ph is None:
        return None
    return (ph[0] or ""), ph[1]


def load_t5(device):
    from transformers import AutoTokenizer, T5EncoderModel

    tokenizer = AutoTokenizer.from_pretrained(T5_MODEL)
    encoder = T5EncoderModel.from_pretrained(T5_MODEL).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return tokenizer, encoder


@torch.no_grad()
def encode_phrases(phrases: Sequence[str], device, batch_size: int = 64, t5=None) -> Dict[str, torch.Tensor]:
    """Mean-pooled T5 encoder vector (d_text,) per phrase, returned on CPU."""
    tokenizer, encoder = t5 if t5 is not None else load_t5(device)
    phrases = sorted(set(phrases))
    out: Dict[str, torch.Tensor] = {}
    for start in range(0, len(phrases), batch_size):
        batch = phrases[start:start + batch_size]
        enc = tokenizer(batch, max_length=MAX_TEXT_LEN, padding="max_length", truncation=True,
                        return_tensors="pt")
        ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        hidden = encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        w = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
        for p, v in zip(batch, pooled.cpu()):
            out[p] = v
    return out


@torch.no_grad()
def encode_spans(queries: Sequence[str], dataset: str, device, batch_size: int = 64,
                 verbose: bool = True) -> Dict[str, torch.Tensor]:
    """Span table ``{query: (4, d_text) tensor}`` for all queries (unmatched queries → zeros)."""
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {DATASETS}")
    queries = sorted(set(queries))
    parsed = {q: span_phrases(q, dataset) for q in queries}
    phrases = sorted({p for ph in parsed.values() if ph is not None for p in ph if p is not None})
    n_unmatched = sum(1 for ph in parsed.values() if ph is None)
    if verbose:
        print(f"  [text] T5-base spans for {len(queries)} queries ({len(phrases)} phrases, "
              f"{n_unmatched} unmatched) …", flush=True)
    vec = encode_phrases(phrases, device, batch_size)
    d_text = next(iter(vec.values())).shape[0] if vec else 768
    zero = torch.zeros(d_text)
    table = {}
    for q, ph in parsed.items():
        if ph is None:
            table[q] = torch.zeros(N_CHANNELS, d_text)
        else:
            table[q] = torch.stack([vec[p] if p is not None else zero for p in ph], dim=0)
    return table
