import os
import signal
import traceback
from tempfile import TemporaryDirectory
import argparse

import numpy as np
import torch
import yaml
from torch.utils.tensorboard._convert_np import make_np
from torch.utils.tensorboard.summary import _calc_scale_factor
from torch.utils.tensorboard._utils import convert_to_HWC

def prepare_image_tensor(tensor, dataformats="NCHW"):
    tensor = make_np(tensor)
    tensor = convert_to_HWC(tensor, dataformats)
    # Do not assume that user passes in values in [0, 255], use data type to detect
    scale_factor = _calc_scale_factor(tensor)
    tensor = tensor.astype(np.float32)
    tensor = (tensor * scale_factor).astype(np.uint8)
    return tensor

def write_image_tensor(prefix: str, tensor: np.ndarray):
    from PIL import Image

    image = Image.fromarray(tensor)
    filename = prefix + ".png"
    image.save(filename, format="png")
    print(f"Image saved to {filename}")
    return filename

def npy_to_png(npy_path, output_prefix=None):
    """
    Convert a .npy file to a .png image
    
    Args:
        npy_path (str): Path to the .npy file
        output_prefix (str, optional): Prefix for the output PNG file. If None, uses the same name as input.
    
    Returns:
        str: Path to the saved PNG file
    """
    # Load the numpy array
    tensor = np.load(npy_path)
    
    # Determine the output path prefix if not provided
    if output_prefix is None:
        output_prefix = os.path.splitext(npy_path)[0]
    
    # Handle different array formats
    if len(tensor.shape) == 2:  # HW
        # Already in format suitable for PIL
        pass
    elif len(tensor.shape) == 3 and tensor.shape[2] in [1, 3, 4]:  # HWC
        # Already in HWC format
        pass
    elif len(tensor.shape) == 3 and tensor.shape[0] in [1, 3, 4]:  # CHW
        # Using the existing function to convert to HWC
        tensor = prepare_image_tensor(tensor, "CHW")
    elif len(tensor.shape) == 4 and tensor.shape[0] == 1:  # NCHW with batch size 1
        tensor = prepare_image_tensor(tensor, "NCHW")
    else:
        raise ValueError(f"Cannot interpret array shape {tensor.shape} as an image")
    
    # Save the image using the existing function
    return write_image_tensor(output_prefix, tensor)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert NumPy arrays (.npy files) to PNG images")
    parser.add_argument("npy_path", help="Path to the .npy file to convert")
    parser.add_argument("-o", "--output", help="Output prefix for the PNG file (optional)")
    
    args = parser.parse_args()
    npy_to_png(args.npy_path, args.output)