"""Utility functions used for feature extractors."""
import abc
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn

import ocl.typing


class FeatureExtractor(nn.Module, metaclass=abc.ABCMeta):
    """Abstract base class for Feature Extractors.

    We expect that the forward method returns a flattened representation of the features, to make
    outputs consistent and not dependent on equal spacing or the dimensionality of the spatial
    information.
    """

    @abc.abstractmethod
    def forward(self, inputs: ocl.typing.ImageOrVideoFeatures) -> ocl.typing.FeatureExtractorOutput:
        pass


class ImageFeatureExtractor(FeatureExtractor):
    """Base class that allows operation of image based feature extractors on videos.

    This is implemented by reshaping the frame dimesion into the batch dimension and
    inversing the process after extraction of the features.

    Subclasses override the `forward_images` method.
    """

    @abc.abstractmethod
    def forward_images(
        self, images: ocl.typing.ImageData
    ) -> Union[
        Tuple[ocl.typing.ImageFeatures, ocl.typing.Positions],
        Tuple[ocl.typing.ImageFeatures, ocl.typing.Positions, Dict],
    ]:
        """Apply feature extractor to image tensor.

        Returns:
            - `torch.Tensor` of extracted features
            - `torch.Tensor` of spatial positions of extracted features
            - Optional dict with additional auxilliary features or information
                from the feature extractor.
        """

    def forward(self, video: ocl.typing.ImageOrVideoData) -> ocl.typing.FeatureExtractorOutput:
        """Apply subclass image feature extractor to potential video data.

        Args:
            video: 5D tensor for video data or 4D tensor for image data.

        Returns:
            ocl.typing.FeatureExtractorOutput: The extracted features with positiional information
                and potential auxilliary features.
        """
        ndim = video.dim()
        assert ndim == 4 or ndim == 5

        if ndim == 5:
            # Handling video data.
            bs, frames, channels, height, width = video.shape
            images = video.view(bs * frames, channels, height, width).contiguous()
        else:
            images = video

        result = self.forward_images(images)

        if isinstance(result, (Tuple, List)):
            if len(result) == 2:
                features, positions = result
                aux_features = None
            elif len(result) == 3:
                features, positions, aux_features = result
            else:
                raise RuntimeError("Expected either 2 or 3 element tuple from `forward_images`.")
        else:
            # Assume output is simply a tensor without positional information.
            return ocl.typing.FeatureExtractorOutput(result, None, None)

        if ndim == 5:
            features = features.unflatten(0, (bs, frames))
            if aux_features is not None:
                aux_features = {k: f.unflatten(0, (bs, frames)) for k, f in aux_features.items()}

        return ocl.typing.FeatureExtractorOutput(features, positions, aux_features)


def cnn_compute_positions_and_flatten(
    features: ocl.typing.CNNImageFeatures,
) -> Tuple[ocl.typing.ImageFeatures, ocl.typing.Positions]:
    """Flatten CNN features to remove spatial dims and return them with correspoding positions."""
    spatial_dims = features.shape[2:]
    positions = torch.cartesian_prod(
        *[torch.linspace(0.0, 1.0, steps=dim, device=features.device) for dim in spatial_dims]
    )
    # reorder into format (batch_size, flattened_spatial_dims, feature_dim).
    flattened = torch.permute(features.view(features.shape[:2] + (-1,)), (0, 2, 1)).contiguous()
    return flattened, positions


def _infer_spatial_dims_from_tokens(
    n_tokens: int, image_size: Optional[Tuple[int, int]] = None
) -> Tuple[int, int]:
    """Infer a 2D grid (H, W) for a sequence of tokens.

    When ``image_size`` is provided, we choose a factorization of ``n_tokens`` whose aspect ratio
    best matches the original image's aspect ratio. This allows handling non-square inputs such as
    COCO images for ViT-style models.

    If ``image_size`` is omitted, we infer a grid solely from ``n_tokens`` by choosing the factor
    pair (H, W) with aspect ratio closest to 1.0 (i.e. as square as possible). This relaxes the
    historical requirement that the number of tokens must form a perfect square while remaining
    backwards compatible for models that do use square inputs.
    """
    if image_size is not None:
        height, width = image_size
        target_aspect = float(height) / float(width)
    else:
        # Prefer an approximately square grid when we don't know the original image size.
        target_aspect = 1.0

    best_dims: Optional[Tuple[int, int]] = None
    best_diff: Optional[float] = None

    limit = int(math.sqrt(n_tokens))
    for h in range(1, limit + 1):
        if n_tokens % h != 0:
            continue
        w = n_tokens // h

        for H, W in ((h, w), (w, h)):
            ratio = float(H) / float(W)
            diff = abs(ratio - target_aspect)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_dims = (H, W)

    if best_dims is None:
        # Fallback to the closest square-ish grid.
        H = int(round(math.sqrt(n_tokens)))
        W = n_tokens // H
        assert (
            H * W == n_tokens
        ), "Could not infer spatial dimensions for Transformer features; please use square images."
        return H, W

    return best_dims


def transformer_compute_positions(
    features: ocl.typing.TransformerImageFeatures,
    image_size: Optional[Tuple[int, int]] = None,
) -> ocl.typing.Positions:
    """Compute positions for Transformer features.

    Args:
        features: Transformer features of shape (batch, tokens, dim).
        image_size: Optional tuple (height, width) of the original input image. When provided,
            positions are inferred for a potentially non-square token grid whose aspect ratio
            matches the image. When omitted, we assume a square grid as in the original
            implementation.
    """
    n_tokens = features.shape[1]
    height_tokens, width_tokens = _infer_spatial_dims_from_tokens(n_tokens, image_size)

    spatial_dims = (height_tokens, width_tokens)
    positions = torch.cartesian_prod(
        *[torch.linspace(0.0, 1.0, steps=dim, device=features.device) for dim in spatial_dims]
    )
    return positions
