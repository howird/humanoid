import torch
import numpy as np
from typing import Tuple, List, Dict, Any

from pycocotools import mask as mask_utils
from humanoid_vision.utils.utils_dataset import process_image, process_mask


def get_cropped_image(
    image: np.ndarray, bbox: np.ndarray, bbox_pad: np.ndarray, seg_mask: np.ndarray
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, List[Dict[str, Any]], np.ndarray, np.ndarray]:
    """Process an image and a single instance's bounding box and segmentation mask.

    1. Encodes the segmentation mask using RLE (Run-Length Encoding)
    2. Processes the mask and image according to the bounding box coordinates
    3. Returns the processed image and associated geometric information

    Args:
        image: Input image array of shape (H, W, 3)
        bbox: Bounding box coordinates [x1, y1, x2, y2] of shape (4,)
        bbox_pad: Padded bounding box coordinates [x1, y1, x2, y2] of shape (4,)
        seg_mask: Binary segmentation mask of shape (H, W)

    Returns:
        masked_image: Processed image tensor with mask channel of shape (4, H', W')
        center_: Center coordinates of the original bounding box of shape (2,)
        scale_: Scale factors of the original bounding box of shape (2,)
        rles: List of run-length encoded segmentation masks, each containing 'counts' and 'size' keys
        center_pad: Center coordinates of the padded bounding box of shape (2,)
        scale_pad: Scale factors of the padded bounding box of shape (2,)
    """
    # Encode the mask for storing, borrowed from tao dataset
    # https://github.com/TAO-Dataset/tao/blob/master/scripts/detectors/detectron2_infer.py
    masks_decoded = np.array(np.expand_dims(seg_mask, 2), order="F", dtype=np.uint8)
    rles = mask_utils.encode(masks_decoded)
    for rle in rles:
        rle["counts"] = rle["counts"].decode("utf-8")

    seg_mask = seg_mask.astype(int) * 255
    if len(seg_mask.shape) == 2:
        seg_mask = np.expand_dims(seg_mask, 2)
        seg_mask = np.repeat(seg_mask, 3, 2)

    center_ = np.array([(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2])
    scale_ = np.array([(bbox[2] - bbox[0]), (bbox[3] - bbox[1])])

    center_pad = np.array([(bbox_pad[2] + bbox_pad[0]) / 2, (bbox_pad[3] + bbox_pad[1]) / 2])
    scale_pad = np.array([(bbox_pad[2] - bbox_pad[0]), (bbox_pad[3] - bbox_pad[1])])
    mask_tmp = process_mask(seg_mask.astype(np.uint8), center_pad, 1.0 * np.max(scale_pad))
    image_tmp = process_image(image, center_pad, 1.0 * np.max(scale_pad))

    masked_image = torch.cat((image_tmp, mask_tmp[:1, :, :]), 0)

    return masked_image, center_, scale_, rles, center_pad, scale_pad
