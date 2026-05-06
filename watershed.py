# 文件：algorithms/watershed.py
import sys
import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)  # insert(0) 优先查找本地模块

import numpy as np
import cv2
from skimage.segmentation import watershed
from preprocess import preprocess_image
from utils import solidity_based_seed_marking


def watershed_segmentation(binary_image, markers):
    dist = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
    labels = watershed(-dist, markers, mask=binary_image)
    return labels

def segment_watershed(image_path):
    gray, sharp, binary = preprocess_image(image_path)
    dist_vis = cv2.normalize(cv2.distanceTransform(binary, cv2.DIST_L2, 5), None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8)
    markers = solidity_based_seed_marking(binary, dist_vis, solidity_threshold=0.7)
    labels = watershed_segmentation(binary, markers)

    # === 提取预测轮廓
    pred_mask = (labels > 0).astype(np.uint8)
    pred_contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return pred_contours