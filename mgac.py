# 文件：algorithms/mcv.py
import sys
import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)  # insert(0) 优先查找本地模块
import numpy as np
import cv2
from skimage.segmentation import morphological_geodesic_active_contour as mgac
from preprocess import preprocess_image
from utils import auto_generate_init_mask_from_distance


def apply_mgac(binary, init_mask, iter=100, balloon=4, gamma=0.7):
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    g = dist / (dist.max() + 1e-5)
    gimg = np.power(g, gamma)
    result = mgac(gimg, num_iter=iter, init_level_set=init_mask,
                  smoothing=4, threshold=0.5, balloon=balloon)
    return result.astype(np.uint8)

def segment_mgac(image_path):

    gray, sharp, binary = preprocess_image(image_path)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    # MGAC流水
    init =auto_generate_init_mask_from_distance(dist, binary, solidity_threshold=0.75)
    seg = apply_mgac(binary, init)

    pred_mask = (seg == 1).astype(np.uint8)
    pred_contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return pred_contours