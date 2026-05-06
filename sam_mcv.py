# 文件：algorithms/sam_mgac.py
import sys
import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)  # insert(0) 优先查找本地模块

import cv2
import numpy as np
from tqdm import tqdm
import torch
from multiprocessing import Pool, cpu_count
from skimage.segmentation import morphological_chan_vese
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from preprocess import preprocess_image
from utils import (
    remove_containing_masks,
    remove_small_sparse_regions,
    show_step,
    auto_generate_init_mask_from_distance,
)
from visualization import visualize_masks_on_image
import inspect


# ✅ 包装函数，兼容 scikit-image 新旧版本
def morphological_chan_vese_safe(image, num_iter, init_levelset, smoothing, lambda1, lambda2):
    sig = inspect.signature(morphological_chan_vese)
    if "init_levelset" in sig.parameters:
        # 旧版本 (<=0.19)
        return morphological_chan_vese(
            image,
            num_iter=num_iter,
            init_levelset=init_levelset,
            smoothing=smoothing,
            lambda1=lambda1,
            lambda2=lambda2,
        )
    else:
        # 新版本 (>=0.20)
        return morphological_chan_vese(
            image,
            num_iter=num_iter,
            init_level_set=init_levelset,
            smoothing=smoothing,
            lambda1=lambda1,
            lambda2=lambda2,
        )


def run_single_mgac(args):
    cnt, gimg, iters, smooth, lambda1, lambda2, shape = args
    level = np.zeros(shape, dtype=np.int8)
    cv2.drawContours(level, [cnt], -1, 1, -1)

    for _ in range(iters):
        level = morphological_chan_vese_safe(
            gimg,
            num_iter=1,
            init_levelset=level.astype(bool),
            smoothing=smooth,
            lambda1=lambda1,
            lambda2=lambda2,
        )
    return level > 0, None


def run_sam_plus_mcv(
    image_path,
    sam_model_path,
    sam_thresh=0.75,
    solidity_thresh=0.8,
    mgac_iters=20,
    mgac_smooth=1,
    mgac_lambda1=1,
    mgac_lambda2=2,
    mgac_gamma=1.0,
    max_rounds=2,
):
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray, sharp, binary = preprocess_image(image_path)
    H, W = gray.shape

    # === SAM 生成掩码 ===
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_h"](checkpoint=sam_model_path).to(device)
    generator = SamAutomaticMaskGenerator(sam)
    anns_raw = generator.generate(img_rgb)
    anns = remove_containing_masks(anns_raw)

    # 初始化最终接受的 mask 和需 refine 的 mask 合并图
    final_mask_sam_valid = np.zeros((H, W), dtype=np.uint8)
    all_sam_mask = np.zeros((H, W), dtype=np.uint8)
    sam_mask = np.zeros((H, W), dtype=np.uint8)

    contour_storage = {
        "sam_valid": [],
        "sam_invalid": [],
        "refined_sam_invalid": [],
        "mgac_rounds": [],
    }

    solidity_strict_thresh = 0.95

    for ann in anns:
        mask = ann["segmentation"]
        mask_area = mask.sum()
        stability = ann.get("stability_score", 1.0)

        # 过滤过小、过大、unstable 区域
        if mask_area < 10 or mask_area > 0.05 * H * W or stability < sam_thresh:
            continue

        mask_bin = mask.astype(np.uint8)
        all_sam_mask = cv2.bitwise_or(all_sam_mask, mask_bin)
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        area = cv2.contourArea(cnt)
        solidity = area / (hull_area + 1e-5)

        if solidity >= solidity_strict_thresh:
            final_mask_sam_valid[mask] = 1
            contour_storage["sam_valid"].append(cnt)
        else:
            contour_storage["sam_invalid"].append(cnt)

        sam_mask[mask] = 1

    merged_mask = cv2.bitwise_and(all_sam_mask, cv2.bitwise_not(final_mask_sam_valid))
    merged_mask = remove_small_sparse_regions(merged_mask, area_thresh=100, fill_thresh=0.1)

    visualize_masks_on_image(
        img_rgb, final_mask_sam_valid, merged_mask, output_path="outputs/sam_mgac/overlay_visualization.png"
    )

    # === 对SAM不稳定区域进行MGAC refine ===
    final_mask_sam_refined = np.zeros((H, W), dtype=bool)

    if np.count_nonzero(merged_mask) > 0:
        masked_binary = cv2.bitwise_and(binary, binary, mask=merged_mask.astype(np.uint8) * 255)
        dist = cv2.distanceTransform(masked_binary, cv2.DIST_L2, 5)
        gimg = (dist / (dist.max() + 1e-5)) ** mgac_gamma
        init = auto_generate_init_mask_from_distance(dist, binary, solidity_thresh)
        cnts, _ = cv2.findContours((init > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        args = [(c, gimg, mgac_iters, mgac_smooth, mgac_lambda1, mgac_lambda2, init.shape) for c in cnts]
        with Pool(min(cpu_count(), 8)) as p:
            results = list(tqdm(p.imap(run_single_mgac, args), total=len(args)))
        for r, _ in results:
            final_mask_sam_refined |= r
            mask = (r.astype(np.uint8)) * 255
            cnts_ref, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_storage["refined_sam_invalid"].extend(cnts_ref)

    # === Binary 其他部分处理 ===
    overall = np.zeros((H, W), dtype=bool)
    refine_round_masks = []
    contour_storage["mgac_rounds"] = []
    combined_sam = np.maximum(final_mask_sam_valid, final_mask_sam_refined.astype(np.uint8))

    for rnd in range(max_rounds):
        rest = cv2.bitwise_and(binary, binary, mask=(1 - sam_mask).astype(np.uint8) * 255)
        rest = cv2.bitwise_and(rest, rest, mask=(~overall).astype(np.uint8) * 255)
        clean_rest = remove_small_sparse_regions(rest, area_thresh=100, fill_thresh=0.2).astype(np.uint8) * 255

        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        gimg = (dist / (dist.max() + 1e-5)) ** mgac_gamma
        init = auto_generate_init_mask_from_distance(dist, clean_rest, solidity_thresh)
        cnts, _ = cv2.findContours((init > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            break

        args = [(c, gimg, 20, 1, 1, 2, rest.shape) for c in cnts]
        with Pool(min(cpu_count(), 8)) as p:
            res = list(tqdm(p.imap(run_single_mgac, args), total=len(args)))

        this_round_contours = []
        for r, _ in res:
            mask_r = (r.astype(np.uint8)) * 255
            cnts_r, _ = cv2.findContours(mask_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            this_round_contours.extend(cnts_r)
            overall |= r

        refine_round_masks.append(overall.copy())
        contour_storage["mgac_rounds"].append(this_round_contours)

    return contour_storage
