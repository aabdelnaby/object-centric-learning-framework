"""Implementation of datasets."""
import collections
import csv
import json
import logging
import os
from distutils.util import strtobool
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import braceexpand
import numpy as np
import pytorch_lightning as pl
import torch
import torchdata
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data._utils import collate as torch_collate
from torchdata.datapipes.iter import IterDataPipe

from torchvision import transforms as tv_transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tvF

import ocl.utils.dataset_patches  # noqa: F401
from ocl.data_decoding import default_decoder
from ocl.transforms import Transform

LOGGER = logging.getLogger(__name__)
USE_AWS_SDK = strtobool(os.getenv("USE_AWS_SDK", "True"))


def _filter_keys(d: dict, keys_to_keep: Tuple) -> Dict[str, Any]:
    """Filter dict for keys in keys_to_keep.

    Additionally keeps all keys which start with `_`.

    Args:
        d: Dict to filter
        keys_to_keep: prefixes used to filter keys in dict.
            Keys with a matching prefix in `keys_to_keep` will be kept.

    Returns:
        The filtered dict.
    """
    keys_to_keep = ("_",) + keys_to_keep
    return {
        key: value
        for key, value in d.items()
        if any(key.startswith(prefix) for prefix in keys_to_keep)
    }


def _get_batch_transforms(transforms: Sequence[Transform]) -> Tuple[Transform]:
    return tuple(filter(lambda t: t.is_batch_transform, transforms))


def _get_single_element_transforms(transforms: Sequence[Transform]) -> Tuple[Transform]:
    return tuple(filter(lambda t: not t.is_batch_transform, transforms))


def _collect_fields(transforms: Sequence[Transform]) -> Tuple[str]:
    return tuple(chain.from_iterable(transform.fields for transform in transforms))


def _get_sorted_values(transforms: Dict[str, Transform]) -> Tuple[Transform]:
    return tuple(transforms[key] for key in sorted(transforms.keys()))


class WebdatasetDataModule(pl.LightningDataModule):
    """Webdataset Data Module."""

    def __init__(
        self,
        train_shards: Optional[Union[str, List[str]]] = None,
        val_shards: Optional[Union[str, List[str]]] = None,
        test_shards: Optional[Union[str, List[str]]] = None,
        batch_size: int = 32,
        eval_batch_size: Optional[int] = None,
        train_transforms: Optional[Dict[str, Transform]] = None,
        eval_transforms: Optional[Dict[str, Transform]] = None,
        num_workers: int = 2,
        train_size: Optional[int] = None,
        val_size: Optional[int] = None,
        test_size: Optional[int] = None,
        shuffle_train: bool = True,
        shuffle_buffer_size: Optional[int] = None,
        use_autopadding: bool = False,
    ):
        """Initialize WebdatasetDataModule.

        Args:
            train_shards: Shards associated with training split. Supports braceexpand notation.
            val_shards: Shards associated with validation split. Supports braceexpand notation.
            test_shards: Shards associated with test split. Supports braceexpand notation.
            batch_size: Batch size to use for training.
            eval_batch_size: Batch size to use for evaluation (i.e. on validation and test split).
                If `None` use same value as during training.
            train_transforms: Transforms to apply during training. We use a dict here to make
                composition of configurations with hydra more easy.
            eval_transforms: Transforms to apply during evaluation. We use a dict here to make
                composition of configurations with hydra more easy.
            num_workers: Number of workers to run in parallel.
            train_size: Number of instance in the train split (used for progressbar).
            val_size: Number of instance in the validation split (used for progressbar).
            test_size: Number of instance in the test split (used for progressbar).
            shuffle_train: Shuffle training split. Only used to speed up operations on train split
                unrelated to training. Should typically be left at `False`.
            shuffle_buffer_size: Buffer size to use for shuffling. If `None` uses `4*batch_size`.
            use_autopadding: Enable autopadding of instances with different dimensions.
        """
        super().__init__()
        if shuffle_buffer_size is None:
            # Ensure that data is shuffled umong at least 4 batches.
            # This should give us a good amount of diversity while also
            # ensuring that we don't need to long to start training.
            # TODO: Ideally, this should also take into account that
            # dataset might be smaller that the shuffle buffer size.
            # As this should not typically occur and we cannot know
            # the number of workers ahead of time we ignore this for now.
            shuffle_buffer_size = batch_size * 4

        if train_shards is None and val_shards is None and test_shards is None:
            raise ValueError("No split was specified. Need to specify at least one split.")
        self.train_shards = train_shards
        self.val_shards = val_shards
        self.test_shards = test_shards
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size if eval_batch_size is not None else batch_size
        self.num_workers = num_workers
        self.shuffle_train = shuffle_train
        self.shuffle_buffer_size = shuffle_buffer_size
        self.train_transforms = _get_sorted_values(train_transforms) if train_transforms else []
        self.eval_transforms = _get_sorted_values(eval_transforms) if eval_transforms else []

        if use_autopadding:
            self.collate_fn = collate_with_autopadding
        else:
            self.collate_fn = collate_with_batch_size

    def _create_webdataset(
        self,
        uri_expression: Union[str, List[str]],
        shuffle=False,
        n_datapoints: Optional[int] = None,
        keys_to_keep: Tuple[str] = tuple(),
        transforms: Sequence[Callable[[IterDataPipe], IterDataPipe]] = tuple(),
    ):
        if isinstance(uri_expression, str):
            uri_expression = [uri_expression]
        # Get shards for current worker.
        shard_uris = list(
            chain.from_iterable(
                braceexpand.braceexpand(single_expression) for single_expression in uri_expression
            )
        )
        datapipe = torchdata.datapipes.iter.IterableWrapper(shard_uris, deepcopy=False)
        datapipe = datapipe.shuffle(buffer_size=len(shard_uris)).sharding_filter()

        if shard_uris[0].startswith("s3://") and USE_AWS_SDK:
            # S3 specific implementation is much faster than fsspec.
            datapipe = datapipe.load_files_by_s3()
        else:
            datapipe = datapipe.open_files_by_fsspec(mode="rb")

        datapipe = datapipe.load_from_tar().webdataset()
        # Discard unneeded properties of the elements prior to shuffling and decoding.
        datapipe = datapipe.map(partial(_filter_keys, keys_to_keep=keys_to_keep))

        if shuffle:
            datapipe = datapipe.shuffle(buffer_size=self.shuffle_buffer_size)

        # Decode files and remove extensions from input as we already decoded the elements. This
        # makes our pipeline invariant to the exact encoding used in the dataset.
        datapipe = datapipe.map(default_decoder)

        # Apply element wise transforms.
        for transform in transforms:
            datapipe = transform(datapipe)
        return torchdata.datapipes.iter.LengthSetter(datapipe, n_datapoints)

    def _create_dataloader(self, dataset, batch_transforms, size, batch_size, partial_batches):
        # Don't return partial batches during training as these give the partial samples a higher
        # weight in the optimization than the other samples of the dataset.

        # Apply batch transforms.
        dataset = dataset.batch(
            batch_size,
            drop_last=not partial_batches,
        ).collate(collate_fn=self.collate_fn)

        for transform in batch_transforms:
            dataset = transform(dataset)

        dataloader = DataLoader(dataset, num_workers=self.num_workers, batch_size=None)

        return dataloader

    def train_data_iterator(self):
        if self.train_shards is None:
            raise ValueError("Can not create train_data_iterator. No training split was specified.")
        transforms = self.train_transforms
        return self._create_webdataset(
            self.train_shards,
            shuffle=self.shuffle_train,
            n_datapoints=self.train_size,
            keys_to_keep=_collect_fields(transforms),
            transforms=_get_single_element_transforms(transforms),
        )

    def train_dataloader(self):
        return self._create_dataloader(
            dataset=self.train_data_iterator(),
            batch_transforms=_get_batch_transforms(self.train_transforms),
            size=self.train_size,
            batch_size=self.batch_size,
            partial_batches=False,
        )

    def val_data_iterator(self):
        if self.val_shards is None:
            raise ValueError("Can not create val_data_iterator. No val split was specified.")
        transforms = self.eval_transforms
        return self._create_webdataset(
            self.val_shards,
            shuffle=False,
            n_datapoints=self.val_size,
            keys_to_keep=_collect_fields(transforms),
            transforms=_get_single_element_transforms(transforms),
        )

    def val_dataloader(self):
        return self._create_dataloader(
            dataset=self.val_data_iterator(),
            batch_transforms=_get_batch_transforms(self.eval_transforms),
            size=self.val_size,
            batch_size=self.eval_batch_size,
            partial_batches=True,
        )

    def test_data_iterator(self):
        if self.test_shards is None:
            raise ValueError("Can not create test_data_iterator. No test split was specified.")
        return self._create_webdataset(
            self.test_shards,
            shuffle=False,
            n_datapoints=self.test_size,
            keys_to_keep=_collect_fields(self.eval_transforms),
            transforms=_get_single_element_transforms(self.eval_transforms),
        )

    def test_dataloader(self):
        return self._create_dataloader(
            dataset=self.test_data_iterator(),
            batch_transforms=_get_batch_transforms(self.eval_transforms),
            size=self.test_size,
            batch_size=self.eval_batch_size,
            partial_batches=True,
        )


class DummyDataModule(pl.LightningDataModule):
    """Dataset providing dummy data for testing."""

    def __init__(
        self,
        data_shapes: Dict[str, List[int]],
        data_types: Dict[str, str],
        batch_size: int = 4,
        eval_batch_size: Optional[int] = None,
        train_transforms: Optional[Dict[str, Transform]] = None,
        eval_transforms: Optional[Dict[str, Transform]] = None,
        train_size: Optional[int] = None,
        val_size: Optional[int] = None,
        test_size: Optional[int] = None,
        # Remaining args needed for compatibility with other datamodules
        train_shards: Optional[str] = None,
        val_shards: Optional[str] = None,
        test_shards: Optional[str] = None,
        num_workers: Optional[int] = None,
    ):
        """Initialize DummyDataModule.

        Args:
            data_shapes: Mapping field names to shapes of tensors.
            data_types: Mapping from field names to types of tensors. One of `image`, `binary`,
                `uniform`, `categorical_{n_categories}` or `mask`.
            batch_size: Batch size to use for training.
            eval_batch_size: Batch size to use for evaluation (i.e. on validation and test split).
                If `None` use same value as during training.
            train_transforms: Transforms to apply during training.
            eval_transforms: Transforms to apply during evaluation.
            train_size: Number of instance in the train split (used for progressbar).
            val_size: Number of instance in the validation split (used for progressbar).
            test_size: Number of instance in the test split (used for progressbar).
            train_shards: Kept for compatibility with WebdatasetDataModule. Has no effect.
            val_shards: Kept for compatibility with WebdatasetDataModule. Has no effect.
            test_shards: Kept for compatibility with WebdatasetDataModule. Has no effect.
            num_workers: Kept for compatibility with WebdatasetDataModule. Has no effect.
        """
        super().__init__()
        self.data_shapes = {key: list(shape) for key, shape in data_shapes.items()}
        self.data_types = data_types
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size if eval_batch_size is not None else batch_size
        self.train_transforms = _get_sorted_values(train_transforms) if train_transforms else []
        self.eval_transforms = _get_sorted_values(eval_transforms) if eval_transforms else []

        self.train_size = train_size
        if self.train_size is None:
            self.train_size = 3 * batch_size + 1
        self.val_size = val_size
        if self.val_size is None:
            self.val_size = 2 * batch_size
        self.test_size = test_size
        if self.test_size is None:
            self.test_size = 2 * batch_size

    @staticmethod
    def _get_random_data_for_dtype(dtype: str, shape: List[int]):
        if dtype == "image":
            return np.random.randint(0, 256, size=shape, dtype=np.uint8)
        elif dtype == "binary":
            return np.random.randint(0, 2, size=shape, dtype=bool)
        elif dtype == "uniform":
            return np.random.rand(*shape).astype(np.float32)
        elif dtype.startswith("categorical_"):
            bounds = [int(b) for b in dtype.split("_")[1:]]
            if len(bounds) == 1:
                lower, upper = 0, bounds[0]
            else:
                lower, upper = bounds
            np_dtype = np.uint8 if upper <= 256 else np.uint64
            return np.random.randint(lower, upper, size=shape, dtype=np_dtype)
        elif dtype.startswith("mask"):
            categories = shape[1]
            np_dtype = np.uint8 if categories <= 256 else np.uint64
            slot_per_pixel = np.random.randint(
                0, categories, size=shape[:1] + shape[2:], dtype=np_dtype
            )
            return (
                np.eye(categories)[slot_per_pixel.reshape(-1)]
                .reshape(shape[:1] + shape[2:] + [categories])
                .transpose([0, 4, 1, 2, 3])
            )
        else:
            raise ValueError(f"Unsupported dtype `{dtype}`")

    def _create_dataset(
        self,
        n_datapoints: int,
        transforms: Sequence[Callable[[Any], Any]],
    ):
        class NumpyDataset(torch.utils.data.IterableDataset):
            def __init__(self, data: Dict[str, np.ndarray], size: int):
                super().__init__()
                self.data = data
                self.size = size

            def __iter__(self):
                for i in range(self.size):
                    elem = {key: value[i] for key, value in self.data.items()}
                    elem["__key__"] = str(i)
                    yield elem

        data = {}
        for key, shape in self.data_shapes.items():
            data[key] = self._get_random_data_for_dtype(self.data_types[key], [n_datapoints] + shape)

        dataset = torchdata.datapipes.iter.IterableWrapper(NumpyDataset(data, n_datapoints))
        for transform in transforms:
            dataset = transform(dataset)

        return dataset

    def _create_dataloader(self, dataset, batch_size):
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=0, collate_fn=collate_with_autopadding
        )

    def train_dataloader(self):
        dataset = self._create_dataset(
            self.train_size, _get_single_element_transforms(self.train_transforms)
        )
        return self._create_dataloader(dataset, self.batch_size)

    def val_dataloader(self):
        dataset = self._create_dataset(
            self.val_size, _get_single_element_transforms(self.eval_transforms)
        )
        return self._create_dataloader(dataset, self.eval_batch_size)

    def test_dataloader(self):
        dataset = self._create_dataset(
            self.test_size, _get_single_element_transforms(self.eval_transforms)
        )
        return self._create_dataloader(dataset, self.eval_batch_size)


class ADE20KSegmentationDataset(torch.utils.data.Dataset):
    """Semantic segmentation dataset for ADE20K in its original folder structure."""

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "training",
        *,
        class_map: Optional[Dict[int, int]] = None,
        class_names: Optional[Dict[int, str]] = None,
        image_transform: Optional[Callable[[Image.Image], Any]] = None,
        mask_transform: Optional[Callable[[Image.Image], Any]] = None,
        sample_limit: Optional[int] = None,
        background_index: int = 0,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.split_dir = self.root / "images" / "ADE" / split
        if not self.split_dir.exists():
            raise ValueError(f"Split directory `{self.split_dir}` does not exist.")

        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.background_index = background_index

        self.class_map, self.class_names = self._init_class_lookup(class_map, class_names)
        self.samples = self._discover_samples(sample_limit)

        self._num_classes = max(self.class_map.values(), default=self.background_index) + 1

    @staticmethod
    def build_class_lookup(
        root: Union[str, Path], background_index: int = 0
    ) -> Tuple[Dict[int, int], Dict[int, str]]:
        """Create mapping from ADE20K name indices to contiguous class ids."""

        objects_file = Path(root) / "objects.txt"
        if not objects_file.exists():
            raise FileNotFoundError(
                f"Could not find `objects.txt` in provided ADE20K root `{root}`."
            )

        class_map: Dict[int, int] = {}
        class_names: Dict[int, str] = {}
        next_index = background_index + 1

        with objects_file.open("r", encoding="utf-8", errors="ignore") as handle:
            reader = csv.reader(handle, delimiter="\t")
            next(reader, None)  # Skip header line.
            for row in reader:
                if len(row) < 2:
                    continue
                try:
                    name_index = int(row[1])
                except ValueError:
                    continue
                if name_index in class_map:
                    continue
                class_map[name_index] = next_index
                class_names[next_index] = row[0]
                next_index += 1

        return class_map, class_names

    def _init_class_lookup(
        self,
        class_map: Optional[Dict[int, int]],
        class_names: Optional[Dict[int, str]],
    ) -> Tuple[Dict[int, int], Dict[int, str]]:
        if class_map is not None:
            if class_names is None:
                class_names = {}
            return class_map, class_names
        return self.build_class_lookup(self.root, self.background_index)

    def _discover_samples(self, sample_limit: Optional[int]) -> List[Dict[str, Path]]:
        samples: List[Dict[str, Path]] = []
        for img_path in sorted(self.split_dir.rglob("*.jpg")):
            stem = img_path.stem
            if stem.endswith("_seg"):
                continue
            annotation_path = img_path.with_suffix(".json")
            segmentation_path = img_path.with_name(f"{stem}_seg.png")
            # Skip auxiliary images inside instance folders.
            if img_path.parent.name == stem:
                continue
            if not annotation_path.exists() or not segmentation_path.exists():
                continue
            samples.append(
                {
                    "image": img_path,
                    "annotation": annotation_path,
                    "segmentation": segmentation_path,
                }
            )
            if sample_limit is not None and len(samples) >= sample_limit:
                break

        if not samples:
            raise RuntimeError(
                f"Found no ADE20K items in `{self.split_dir}`. Check dataset extraction."
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def _load_objects(self, annotation_path: Path) -> List[Dict[str, Any]]:
        with annotation_path.open("r", encoding="utf-8") as handle:
            annotation = json.load(handle)["annotation"]
        objects = annotation.get("object", [])
        if isinstance(objects, dict):
            objects = [objects]
        return objects

    def _build_semantic_mask(
        self, sample: Dict[str, Path]
    ) -> Tuple[np.ndarray, List[int]]:
        segmentation_image = Image.open(sample["segmentation"])
        width, height = segmentation_image.size
        semantic_mask = np.full((height, width), self.background_index, dtype=np.int32)
        present_classes: List[int] = []

        for obj in self._load_objects(sample["annotation"]):
            try:
                name_index = int(obj.get("name_ndx"))
            except (TypeError, ValueError):
                continue

            class_id = self.class_map.get(name_index)
            if class_id is None:
                continue

            mask_rel_path = obj.get("instance_mask")
            if not mask_rel_path:
                continue
            mask_path = self.split_dir / mask_rel_path
            if not mask_path.exists():
                continue

            mask_image = Image.open(mask_path)
            mask_array = np.array(mask_image)
            positive = mask_array == 255
            if not np.any(positive):
                positive = mask_array > 0
            if not np.any(positive):
                continue

            semantic_mask[positive] = class_id
            present_classes.append(class_id)

        return semantic_mask, present_classes

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]

        image = Image.open(sample["image"]).convert("RGB")
        semantic_mask, present_classes = self._build_semantic_mask(sample)

        if self.image_transform:
            image = self.image_transform(image)
        else:
            image = tv_transforms.ToTensor()(image)

        mask_image = Image.fromarray(semantic_mask.astype(np.int32), mode="I")
        if self.mask_transform:
            mask = self.mask_transform(mask_image)
        else:
            mask = tvF.pil_to_tensor(mask_image).squeeze(0).long()

        if isinstance(mask, torch.Tensor):
            mask_tensor = mask.long()
        else:
            mask_tensor = torch.as_tensor(np.array(mask), dtype=torch.long)

        return {
            "image": image,
            "mask": mask_tensor,
            "meta": {
                "image_path": str(sample["image"]),
                "classes": sorted(set(present_classes)),
            },
        }


class ADE20KDataModule(pl.LightningDataModule):
    """Lightning DataModule wrapping :class:`ADE20KSegmentationDataset`."""

    def __init__(
        self,
        data_dir: Union[str, Path],
        *,
        batch_size: int = 4,
        eval_batch_size: Optional[int] = None,
        num_workers: int = 4,
        image_size: int = 224,
        shuffle_buffer_size: Optional[int] = None,
        max_classes: Optional[int] = None,
        train_limit: Optional[int] = None,
        val_limit: Optional[int] = None,
        test_limit: Optional[int] = None,
        pin_memory: bool = False,
        train_image_transform: Optional[Callable[[Image.Image], Any]] = None,
        val_image_transform: Optional[Callable[[Image.Image], Any]] = None,
        mask_transform: Optional[Callable[[Image.Image], Any]] = None,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size if eval_batch_size is not None else batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # The datamodule loads items from disk, but we expose this attribute for API compatibility
        # with the other dataset configs.
        self.shuffle_buffer_size = shuffle_buffer_size
        self.max_classes = max_classes
        self.train_limit = train_limit
        self.val_limit = val_limit
        self.test_limit = test_limit

        self.image_size = image_size

        self.train_image_transform = train_image_transform or self._build_default_image_transform(
            image_size, is_train=True
        )
        self.val_image_transform = val_image_transform or self._build_default_image_transform(
            image_size, is_train=False
        )
        self.mask_transform = mask_transform or self._build_default_mask_transform(image_size)

        self._class_map: Optional[Dict[int, int]] = None
        self._class_names: Optional[Dict[int, str]] = None

        self.train_dataset: Optional[ADE20KSegmentationDataset] = None
        self.val_dataset: Optional[ADE20KSegmentationDataset] = None
        self.test_dataset: Optional[ADE20KSegmentationDataset] = None

    @staticmethod
    def _build_default_image_transform(image_size: int, *, is_train: bool) -> tv_transforms.Compose:
        transforms: List[Any] = []
        if is_train:
            transforms.append(tv_transforms.RandomHorizontalFlip())
        transforms.extend(
            [
                tv_transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
                tv_transforms.ToTensor(),
                tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return tv_transforms.Compose(transforms)

    @staticmethod
    def _build_default_mask_transform(image_size: int) -> tv_transforms.Compose:
        return tv_transforms.Compose(
            [
                tv_transforms.Resize((image_size, image_size), interpolation=InterpolationMode.NEAREST),
                tv_transforms.Lambda(lambda img: tvF.pil_to_tensor(img).squeeze(0).long()),
            ]
        )

    def _ensure_class_lookup(self) -> None:
        if self._class_map is not None:
            return
        self._class_map, self._class_names = ADE20KSegmentationDataset.build_class_lookup(
            self.data_dir
        )
        if self.max_classes is not None:
            filtered_map: Dict[int, int] = {}
            filtered_names: Dict[int, str] = {}
            for name_index, class_id in self._class_map.items():
                if class_id <= self.max_classes:
                    filtered_map[name_index] = class_id
                    if self._class_names and class_id in self._class_names:
                        filtered_names[class_id] = self._class_names[class_id]
            self._class_map = filtered_map
            self._class_names = filtered_names

    @property
    def num_classes(self) -> int:
        if self.train_dataset is not None:
            return self.train_dataset.num_classes
        if self._class_map is None:
            self._ensure_class_lookup()
        assert self._class_map is not None
        return max(self._class_map.values(), default=0) + 1

    def setup(self, stage: Optional[str] = None) -> None:
        self._ensure_class_lookup()

        if stage in (None, "fit"):
            self.train_dataset = ADE20KSegmentationDataset(
                self.data_dir,
                split="training",
                class_map=self._class_map,
                class_names=self._class_names,
                image_transform=self.train_image_transform,
                mask_transform=self.mask_transform,
                sample_limit=self.train_limit,
            )
            self.val_dataset = ADE20KSegmentationDataset(
                self.data_dir,
                split="validation",
                class_map=self._class_map,
                class_names=self._class_names,
                image_transform=self.val_image_transform,
                mask_transform=self.mask_transform,
                sample_limit=self.val_limit,
            )

        if stage in (None, "validate") and self.val_dataset is None:
            self.val_dataset = ADE20KSegmentationDataset(
                self.data_dir,
                split="validation",
                class_map=self._class_map,
                class_names=self._class_names,
                image_transform=self.val_image_transform,
                mask_transform=self.mask_transform,
                sample_limit=self.val_limit,
            )

        test_dir = self.data_dir / "images" / "ADE" / "testing"
        if test_dir.exists() and stage in (None, "test"):
            self.test_dataset = ADE20KSegmentationDataset(
                self.data_dir,
                split="testing",
                class_map=self._class_map,
                class_names=self._class_names,
                image_transform=self.val_image_transform,
                mask_transform=self.mask_transform,
                sample_limit=self.test_limit,
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call `setup('fit')` before requesting the train dataloader.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Call `setup('fit')` or `setup('validate')` before requesting val data.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("Testing split not available or `setup('test')` not called.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


def collate_with_batch_size(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Default pytorch collate function with additional `batch_size` output for dict input."""
    if isinstance(batch[0], collections.abc.Mapping):
        out = torch_collate.default_collate(batch)
        out["batch_size"] = len(batch)
        return out
    return torch_collate.default_collate(batch)


def collate_with_autopadding(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function that takes a batch of data and stacks it with a batch dimension.

    In contrast to torch's collate function, this function automatically pads tensors of different
    sizes with zeros such that they can be stacked.

    Adapted from https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py.
    """
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        # As most tensors will not need padding to be stacked, we first try to stack them normally
        # and pad only if normal padding fails. This avoids explicitly checking whether all tensors
        # have the same shape before stacking.
        try:
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum(x.numel() for x in batch)
                if len(batch) * elem.numel() != numel:
                    # Check whether resizing will fail because tensors have unequal sizes to avoid
                    # a memory allocation. This is a sufficient but not necessary condition, so it
                    # can happen that this check will not trigger when padding is necessary.
                    raise RuntimeError()
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage).resize_(len(batch), *elem.shape)
            return torch.stack(batch, 0, out=out)
        except RuntimeError:
            # Stacking did not work. Try to pad tensors to the same dimensionality.
            if not all(x.ndim == elem.ndim for x in batch):
                raise ValueError("Tensors in batch have different number of dimensions.")

            shapes = [x.shape for x in batch]
            max_dims = [max(shape[idx] for shape in shapes) for idx in range(elem.ndim)]

            paddings = []
            for shape in shapes:
                padding = []
                # torch.nn.functional.pad wants padding from last to first dim, so go in reverse
                for idx in reversed(range(len(shape))):
                    padding.append(0)
                    padding.append(max_dims[idx] - shape[idx])
                paddings.append(padding)

            batch_padded = [
                torch.nn.functional.pad(x, pad, mode="constant", value=0.0)
                for x, pad in zip(batch, paddings)
            ]

            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum(x.numel() for x in batch_padded)
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage).resize_(len(batch_padded), *batch_padded[0].shape)
            return torch.stack(batch_padded, 0, out=out)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if torch_collate.np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(torch_collate.default_collate_err_msg_format.format(elem.dtype))

            return collate_with_autopadding([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        out = {key: collate_with_autopadding([d[key] for d in batch]) for key in elem}
        out["batch_size"] = len(batch)
        try:
            return elem_type(out)
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return out
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(collate_with_autopadding(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [
                collate_with_autopadding(samples) for samples in transposed
            ]  # Backwards compatibility.
        else:
            try:
                return elem_type([collate_with_autopadding(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [collate_with_autopadding(samples) for samples in transposed]

    raise TypeError(torch_collate.default_collate_err_msg_format.format(elem_type))
