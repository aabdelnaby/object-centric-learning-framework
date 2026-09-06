# Data, weights and where they go

Nothing here is in git. `data/`, `checkpoints/`, `runs/`, `results/` and `figures/` are ignored.
On a machine that already holds the original research checkout, `bash tools/link_local_data.sh`
creates all the links below in one go.

## Layout

```
data/
  coco/                      COCO-2017 images (train2017/, val2017/, annotations/)     62 GB
  webdataset/coco2017/       COCO as webdataset shards, only for step 1                49 GB
  paco/
    paco_annotations/        paco_lvis_v1_{train,val,test}.json                       425 MB
    paco_questions.csv       the training file: 50,679 train + 2,656 val questions      13 MB
    paco_val_masks.csv       ground-truth part masks of the validation questions       780 KB
    part_masks/val/*.png     one binary mask per validation question                   213 MB
  cub/
    CUB_200_2011/            the official release (images/, attributes/, splits)       1.1 GB
    cub200_questions.csv     part-attribute questions built from it                     41 MB
  caches/
    dino_paco_square224.pt   frozen patch features, 196×384 fp16 per image             8.4 GB
    dino_cub_square224.pt    the same for CUB                                          1.7 GB
checkpoints/
  dinosaur_dinov3_vits16_coco.ckpt   the pretrained slot module (step 1)               210 MB
  thesis/<model>/best_model.pt       the trained heads reported in the thesis        2-4 MB each
```

## Getting the raw data

**COCO-2017** (images for PACO, and the shards for step 1):

```bash
cd scripts/datasets && bash download_and_convert.sh COCO     # downloads into data/coco, converts to shards
```

**PACO-LVIS** annotations (part masks and colour attributes on COCO images):

```bash
wget https://dl.fbaipublicfiles.com/paco/annotations/paco_lvis_v1.zip
unzip paco_lvis_v1.zip -d data/paco/paco_annotations
```

**CUB-200-2011**: download `CUB_200_2011.tgz` from
<https://www.vision.caltech.edu/datasets/cub_200_2011/> and extract it to `data/cub/CUB_200_2011`.

## Pretrained weights

Downloaded automatically on first use into the Hugging Face cache:

| Weights | Used for | Size |
|---|---|---|
| `timm/vit_small_patch16_dinov3.lvd1689m` | the frozen image backbone | 83 MB |
| `t5-base` | the frozen text encoder (question spans) | 850 MB |
| `OpenGVLab/InternVL3-14B` | the colour fallback in step 2 only | 29 GB |

On compute nodes without internet, download them once on a login node and run the jobs with
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` (step 2 already does this).

The DINOSAUR slot module is trained by `experiments/01_train_dinosaur.sh`; copy the checkpoint you
want to `checkpoints/dinosaur_dinov3_vits16_coco.ckpt`. The thesis used the 11-slot run at 500k
steps. Its conditioning parameters do not depend on the slot count, so the same weights are
instantiated with 9 slots on PACO and 7 on CUB.

## Question sets

Both CSVs share the columns the code reads: `image_name`, `split`, `query`, `label` and, where
present, `label_rank`. Image paths may be relative to the dataset's image root or absolute.

**PACO** (`data/paco/paco_questions.csv`, built by step 2): "What is the color of the `<part>` of
the `<object>`?" for referentially unambiguous object-part pairs, that is a single instance of the
object in the image, a single instance of the part on that object, a part mask of at least 400
pixels, and no typically-repeated part (leg, ear, wheel, side, ...). Answers are 12 basic colours;
the label is PACO's human colour where it exists and an InternVL-3 colour of the part crop
otherwise (82.7 % / 17.3 % on the validation split). Splits: 50,679 train, 2,656 val from 1,080
images.

**CUB** (`data/cub/cub200_questions.csv`, built by step 3): "What is the `<part>` `<attribute>` of
the bird?" from the CUB attribute annotations, dense-ranked by annotator certainty. Training and
evaluation use rank-1 rows and the 16 colour questions (`category_filter=color`), 15 colour
classes, on the official train/test split. CUB has no validation split, so the test split is used
throughout, as in the thesis.
