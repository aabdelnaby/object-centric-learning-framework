"""Image preprocessing and cached DINOv3 patch features.

All models run on frozen patch features, so the ViT is applied once per unique image
(``python -m hier_dinosaur.features``) and the result is cached as
``{"features": {image_name: (196, 384) tensor}, "positions": (196, 2) tensor}``.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from .dinosaur import IMG_SIZE, FrozenDinosaur

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


def _pad_to_square(t: torch.Tensor) -> torch.Tensor:
    c, h, w = t.shape
    m = max(h, w)
    canvas = torch.tensor(IMAGE_MEAN, dtype=t.dtype).view(c, 1, 1).repeat(1, m, m).clone()
    top, left = (m - h) // 2, (m - w) // 2
    canvas[:, top:top + h, left:left + w] = t
    return canvas


def build_image_transform(img_size: int = IMG_SIZE, resize_mode: str = "square"):
    """PIL image → normalised (3, S, S) tensor, bicubic resize (matches the DINOSAUR pretraining).

    ``square`` (thesis) resizes to S×S keeping all content; ``crop`` is the ImageNet-style
    shorter-side resize + centre crop; ``pad`` pads to a square first.
    """
    bicubic = transforms.InterpolationMode.BICUBIC
    clamp = transforms.Lambda(lambda x: x.clamp(0.0, 1.0))
    if resize_mode == "crop":
        geo = [transforms.Resize(img_size, interpolation=bicubic), clamp, transforms.CenterCrop(img_size)]
    elif resize_mode == "square":
        geo = [transforms.Resize((img_size, img_size), interpolation=bicubic), clamp]
    elif resize_mode == "pad":
        geo = [transforms.Lambda(_pad_to_square), transforms.Resize((img_size, img_size), interpolation=bicubic), clamp]
    else:
        raise ValueError(f"unknown resize_mode={resize_mode!r}; expected crop|square|pad")
    return transforms.Compose([transforms.ToTensor()] + geo + [transforms.Normalize(IMAGE_MEAN, IMAGE_STD)])


def resolve_image_path(image_root: str, image_name: str) -> str:
    """Absolute ``image_name`` entries are used as-is, relative ones are joined to ``image_root``."""
    return os.path.join(image_root or "", image_name)


class UniqueImageDataset(Dataset):
    def __init__(self, image_names: List[str], image_root: str, transform):
        self.names, self.root, self.transform = image_names, image_root, transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img = Image.open(resolve_image_path(self.root, name)).convert("RGB")
        return name, self.transform(img)


@torch.no_grad()
def precompute_dino(
    dinosaur: FrozenDinosaur,
    image_names: List[str],
    image_root: str,
    device,
    img_size: int = IMG_SIZE,
    resize_mode: str = "square",
    batch_size: int = 64,
    num_workers: int = 4,
    fp16: bool = True,
) -> Dict:
    ds = UniqueImageDataset(image_names, image_root, build_image_transform(img_size, resize_mode))
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=lambda b: ([x[0] for x in b], torch.stack([x[1] for x in b])),
    )
    store_dtype = torch.float16 if fp16 else torch.float32
    dinosaur = dinosaur.to(device)
    features, positions = {}, None
    print(f"  Computing ViT features for {len(image_names)} images at {img_size}px "
          f"(resize_mode={resize_mode}, dtype={store_dtype}) …", flush=True)
    for names, imgs in tqdm(loader, unit="batch"):
        out = dinosaur.encode_images(imgs.to(device))
        feats = out.features.to(store_dtype).cpu()
        if positions is None:
            positions = out.positions.cpu()
        for name, f in zip(names, feats):
            features[name] = f
    return {"features": features, "positions": positions}


def load_dino_cache(path: str) -> Dict:
    return torch.load(path, map_location="cpu")


def main():
    ap = argparse.ArgumentParser(description="Cache frozen DINOv3 patch features for every unique image of a CSV.")
    ap.add_argument("--csv", required=True, help="question CSV with an image_name column")
    ap.add_argument("--image_root", default="", help="prefix for relative image_name entries")
    ap.add_argument("--dinosaur_ckpt", required=True, help="pretrained DINOSAUR .ckpt")
    ap.add_argument("--out", required=True, help="output .pt cache path")
    ap.add_argument("--img_size", type=int, default=IMG_SIZE)
    ap.add_argument("--resize_mode", default="square", choices=["crop", "square", "pad"])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--fp32", action="store_true", help="store float32 instead of float16")
    ap.add_argument("--limit", type=int, default=None, help="only the first N unique images (smoke tests)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing cache")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.force:
        raise SystemExit(f"{args.out} exists; pass --force to rebuild it.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(args.csv)
    names = df["image_name"].unique().tolist()
    if args.limit:
        names = names[:args.limit]
    print(f"Device: {device} | unique images: {len(names)}")
    dinosaur = FrozenDinosaur(n_slots=7, checkpoint=args.dinosaur_ckpt)
    cache = precompute_dino(dinosaur, names, args.image_root, device, args.img_size, args.resize_mode,
                            args.batch_size, args.num_workers, fp16=not args.fp32)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(cache, args.out)
    shape = tuple(next(iter(cache["features"].values())).shape)
    print(f"Saved → {args.out}  ({len(cache['features'])} entries, feature shape {shape})")


if __name__ == "__main__":
    main()
