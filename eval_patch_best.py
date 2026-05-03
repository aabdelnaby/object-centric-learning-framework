"""Eval the best PatchClassifier checkpoint on the per-query test split."""
import sys, os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

CKPT_PATH   = "cub_classifier_checkpoints_patch_control/patches/best_model.pt"
DINO_CFG    = "projects/bridging/dinosaur/coco_feat_rec_dino_small16_auto_dinov3"
DINO_CKPT   = "checkpoints/epoch_67-step_500000_coco.ckpt"
DINO_CACHE  = "FG-datset/CUB_200_2011/dino_feat_cache.pt"
TEXT_CACHE  = "FG-datset/CUB_200_2011/text_feat_cache.pt"
CSV_PATH    = "FG-datset/CUB_200_2011/cub200_ranked_classification_dataset.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt = torch.load(CKPT_PATH, map_location=device)
label_vocab = ckpt["label_vocab"]
num_classes = len(label_vocab)
print(f"Loaded checkpoint  epoch={ckpt['epoch']}  val_acc={ckpt['val_acc']:.4f}  classes={num_classes}")

from classifier_model import PatchClassifier
model = PatchClassifier(
    dinosaur_cfg_name  = DINO_CFG,
    dinosaur_ckpt_path = DINO_CKPT,
    num_classes        = num_classes,
    d_vit              = 384,
    d_slot             = 256,
    d_text             = 1024,
    num_heads          = 8,
    load_text_encoder  = False,
).to(device)

ts = ckpt["trainable_state"]
model.patch_projector.load_state_dict(ts["patch_projector"])
model.text_projector.load_state_dict(ts["text_projector"])
model.gated_cross_attn.load_state_dict(ts["gated_cross_attn"])
model.classifier_head.load_state_dict(ts["classifier_head"])
model.eval()
print("Trainable weights loaded.")

from cub_dataset import CUBCachedFeatDataset

color_queries = [
    "What is the back color of the bird?",
    "What is the belly color of the bird?",
    "What is the bill color of the bird?",
    "What is the breast color of the bird?",
    "What is the crown color of the bird?",
    "What is the eye color of the bird?",
    "What is the forehead color of the bird?",
    "What is the leg color of the bird?",
    "What is the nape color of the bird?",
    "What is the primary color of the bird?",
    "What is the throat color of the bird?",
    "What is the under tail color of the bird?",
    "What is the underparts color of the bird?",
    "What is the upper tail color of the bird?",
    "What is the upperparts color of the bird?",
    "What is the wing color of the bird?",
]

criterion = nn.CrossEntropyLoss()
results = {}

for query in sorted(color_queries):
    ds = CUBCachedFeatDataset(
        csv_path        = CSV_PATH,
        label_vocab     = label_vocab,
        query_filter    = query,
        dino_cache_path = DINO_CACHE,
        text_cache_path = TEXT_CACHE,
        split           = "test",
    )
    if len(ds) == 0:
        continue
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            dino_feat, text_hidden, attn_mask, labels = (x.to(device) for x in batch)
            logits = model.forward_cached(dino_feat, text_hidden, attn_mask)
            correct += (logits.argmax(1) == labels).sum().item()
            total   += labels.size(0)
    acc = correct / total
    results[query] = (acc, total)
    print(f"  {query:<55} acc={acc:.4f}  n={total}")

mean_acc = sum(v[0] for v in results.values()) / len(results)
print(f"\nMean acc: {mean_acc:.4f}")
