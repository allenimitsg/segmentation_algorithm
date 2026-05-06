# 文件：algorithms/mcv.py
import sys
import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import cv2
from skimage.segmentation import morphological_chan_vese
from preprocess import preprocess_image
from utils import auto_generate_init_mask_from_distance


def apply_morphological_chan_vese(binary, init_mask, num_iter=100, smoothing=4, lambda1=1, lambda2=1):
    """
    应用Morphological Chan-Vese水平集分割算法
    """
    # 对于Chan-Vese算法，我们通常直接使用灰度图像或距离变换
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    g = dist / (dist.max() + 1e-5)

    # 检查函数参数并尝试不同的参数名称
    try:
        # 尝试使用 init_levelset
        result = morphological_chan_vese(
            g,
            num_iter=num_iter,
            init_levelset=init_mask,
            smoothing=smoothing,
            lambda1=lambda1,
            lambda2=lambda2
        )
    except TypeError:
        try:
            # 尝试使用 init_level_set
            result = morphological_chan_vese(
                g,
                num_iter=num_iter,
                init_level_set=init_mask,
                smoothing=smoothing,
                lambda1=lambda1,
                lambda2=lambda2
            )
        except TypeError:
            try:
                # 尝试使用 init_level_set 但去掉其他可能不存在的参数
                result = morphological_chan_vese(
                    g,
                    iterations=num_iter,  # 有些版本用 iterations
                    init_level_set=init_mask,
                    smoothing=smoothing
                )
            except TypeError:
                # 最后尝试最基本的调用
                result = morphological_chan_vese(
                    g,
                    iterations=num_iter,
                    init_level_set=init_mask
                )

    return result.astype(np.uint8)


def segment_mcv(image_path, num_iter=100, smoothing=4, lambda1=1, lambda2=2):
    """
    使用Morphological Chan-Vese算法进行图像分割
    保持与原函数相同的接口，返回轮廓列表
    """
    # 预处理图像
    gray, sharp, binary = preprocess_image(image_path)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    # 生成初始掩码
    init_mask = auto_generate_init_mask_from_distance(dist, binary, solidity_threshold=0.75)

    # 应用Morphological Chan-Vese算法
    seg = apply_morphological_chan_vese(
        binary,
        init_mask,
        num_iter=num_iter,
        smoothing=smoothing,
        lambda1=lambda1,
        lambda2=lambda2
    )

    pred_mask = (seg == 1).astype(np.uint8)
    pred_contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return pred_contours


# 兼容性函数，保持与原来相同的接口
def apply_mgac(binary, init_mask, iter=100, balloon=4, gamma=0.7):
    """
    兼容性函数，保持与原来相同的接口
    """
    return apply_morphological_chan_vese(binary, init_mask, num_iter=iter, smoothing=4, lambda1=1, lambda2=2)


# 调试和检查函数
if __name__ == "__main__":
    # 检查函数签名
    import inspect

    print("=== morphological_chan_vese 函数参数 ===")
    sig = inspect.signature(morphological_chan_vese)
    for param_name, param in sig.parameters.items():
        default = param.default if param.default != param.empty else '必需'
        print(f"  {param_name}: {default}")

    print("\n=== 测试调用 ===")
    # 创建一个简单的测试图像
    test_binary = np.zeros((100, 100), dtype=np.uint8)
    test_binary[40:60, 40:60] = 1
    test_init = np.zeros((100, 100), dtype=np.uint8)
    test_init[45:55, 45:55] = 1

    try:
        result = morphological_chan_vese(test_binary, iterations=10, init_level_set=test_init)
        print("成功调用: iterations + init_level_set")
    except TypeError as e:
        print(f"iterations + init_level_set 失败: {e}")

    try:
        result = morphological_chan_vese(test_binary, num_iter=10, init_levelset=test_init)
        print("成功调用: num_iter + init_levelset")
    except TypeError as e:
        print(f"num_iter + init_levelset 失败: {e}")

    try:
        result = morphological_chan_vese(test_binary, iterations=10, init_levelset=test_init)
        print("成功调用: iterations + init_levelset")
    except TypeError as e:
        print(f"iterations + init_levelset 失败: {e}")