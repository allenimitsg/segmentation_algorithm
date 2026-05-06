# 文件：algorithms/sam_only.py
import json
import sys
import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)  # insert(0) 优先查找本地模块

import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from preprocess import preprocess_image
from utils import remove_containing_masks, evaluate_segmentation_from_contours



def sam_only(image_path, sam_checkpoint):

    gray0 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    H, W = gray0.shape

    gray, sharp, morph = preprocess_image(image_path)
    # image_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam = sam_model_registry['vit_h'](checkpoint=sam_checkpoint).to(device)
    generator = SamAutomaticMaskGenerator(sam)
    sharp_rgb = cv2.cvtColor(sharp, cv2.COLOR_GRAY2RGB)
    masks_raw = generator.generate(sharp_rgb)
    masks = remove_containing_masks(masks_raw)

    pred_contours = []
    for m in masks:
        mask = m['segmentation']
        if np.sum(mask) > 0.1 * H * W:
            continue
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pred_contours.extend(cnts)

    return pred_contours