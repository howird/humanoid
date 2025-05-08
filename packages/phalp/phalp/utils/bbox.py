import torch
import numpy as np

from pycocotools import mask as mask_utils
from phalp.utils.utils_dataset import process_image, process_mask


def get_cropped_image(image, bbox, bbox_pad, seg_mask):
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

    # bbox_        = expand_bbox_to_aspect_ratio(bbox, target_aspect_ratio=(192,256))
    # center_x     = np.array([(bbox_[2] + bbox_[0])/2, (bbox_[3] + bbox_[1])/2])
    # scale_x      = np.array([(bbox_[2] - bbox_[0]), (bbox_[3] - bbox_[1])])
    # mask_tmp     = process_mask(seg_mask.astype(np.uint8), center_x, 1.0*np.max(scale_x))
    # image_tmp    = process_image(image, center_x, 1.0*np.max(scale_x))

    masked_image = torch.cat((image_tmp, mask_tmp[:1, :, :]), 0)

    return masked_image, center_, scale_, rles, center_pad, scale_pad
