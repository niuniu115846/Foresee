import sys
import os
sys.path.append('../sam')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
import imageio.v2 as imageio

# import sys
# sys.path.append('../segment_anything')  # 将 segment_anything 目录添加到系统路径中，确保可以导入 segment_anything 模块

# ----------------------------
# 1. 引入 SAM2 构建与预测类
# （以下导入路径以当前项目典型结构为例，若仓库结构变化，请根据实际文件名调整）
from sam.sam2.build_sam import build_sam2
from sam.sam2.sam2_image_predictor import SAM2ImagePredictor
# ----------------------------

def load_image(image_path: str):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb

def polygon_to_box(polygon_pts):
    xs = [p[0] for p in polygon_pts]
    ys = [p[1] for p in polygon_pts]
    return [min(xs), min(ys), max(xs), max(ys)]

# def visualize_and_save(image, mask, save_path="result_overlay.png", alpha=0.55, color=(0, 255, 0)):
#     """
#     image: RGB image (H,W,3)
#     mask:  (H,W) bool or 0/1
#     color: BGR tuple for overlay when converting via cv2, but we will stay in RGB for matplotlib
#     """
#     mask_bool = mask.astype(bool)
#     overlay = image.copy()
#     # 颜色 (R,G,B)
#     overlay[mask_bool] = (
#         0.45 * overlay[mask_bool] + 0.55 * np.array([color[2], color[1], color[0]])
#     ).astype(np.uint8)

#     plt.figure(figsize=(8,8))
#     plt.imshow(overlay)
#     plt.axis("off")
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=150)
#     print(f"Overlay saved to {save_path}")

#     # 保存二值 mask
#     mask_path = Path(save_path).with_suffix(".mask.png")
#     cv2.imwrite(str(mask_path), (mask_bool * 255).astype(np.uint8))
#     print(f"Binary mask saved to {mask_path}")

def denormalize_if_needed(box, W, H, normalized: bool):
    """如果 points 是归一化(0~1)则放大到像素；否则原样返回"""
    if not normalized:
        return box
    if box is None:
        return None
    out = []
    x1,y1,x2,y2 = box
    x1,y1 = x1*W, y1*H
    x2,y2 = x2*W, y2*H
    out = [x1,y1,x2,y2]
    return out

def ensure_points_format(points):
    """统一处理为 [(x,y), ...] 且 float"""
    if points is None:
        return []
    cleaned = []
    for p in points:
        if len(p) != 2:
            raise ValueError(f"Point {p} length != 2")
        cleaned.append((float(p[0]), float(p[1])))
    return cleaned

def print_some_params(model, k=5):
    c = 0
    for name, p in model.named_parameters():
        if p.ndim >= 2:
            print(f"[PARAM] {name} mean={p.mean().item():.6f}, std={p.std().item():.6f}")
            c += 1
            if c >= k: break

def save_soft_prob(
    image: np.ndarray,
    image_path: str,
    logits: np.ndarray,
    scores: np.ndarray,
    best_idx: int,
    output_prefix: str = "./sam2_result",
    hard_threshold: float = 0.5,
    save_all: bool = False,
    cmap: str = "magma",
    overlay_color=(0, 255, 0),
    overlay_alpha: float = 0.55,
    dpi: int = 150,
    verbose: bool = True
):
    """
    保存 SAM2 软概率图及相关可视化结果。

    参数
    ----
    image : 原始 RGB 图 (H,W,3)
    logits: (N, h_l, w_l) 低分辨率未 sigmoid 的 mask logits
    scores: (N,) 每个候选 mask 的得分
    best_idx : 你已经选出来的最佳候选索引
    output_prefix : 输出文件前缀（不带扩展名），例如 "./sam2_result"
    hard_threshold : 将软概率转硬掩码的阈值
    save_all : 是否把所有候选的软概率 & 硬掩码都保存
    cmap : 热力图颜色表
    overlay_color : 概率 / 硬掩码叠加颜色 (R,G,B)
    overlay_alpha : overlay 透明度
    dpi : 保存图片 dpi
    verbose : 是否打印日志

    返回
    ----
    dict 包含保存文件路径及 numpy 数组引用
    """

    # os.makedirs(os.path.dirname(output_prefix) if os.path.dirname(output_prefix) else ".", exist_ok=True)

    if not isinstance(logits, np.ndarray):
        logits = np.asarray(logits)

    N = logits.shape[0]
    H, W = image.shape[:2]

    # 1) logits -> probability (低分辨率 -> 上采样到原图尺寸)
    with torch.no_grad():
        logits_t = torch.from_numpy(logits)  # (N,h_l,w_l)
        prob_low = torch.sigmoid(logits_t)   # (N,h_l,w_l)
        prob_up = F.interpolate(
            prob_low.unsqueeze(1), size=(H, W),
            mode="bilinear", align_corners=False
        ).squeeze(1)                         # (N,H,W)

    # 2) 软概率 -> 硬掩码
    hard_up = (prob_up > hard_threshold).to(torch.uint8)

    # 3) 仅保存 best 或保存全部
    indices = range(N) if save_all else [best_idx]

    saved = {
        "prob_npy": [],
        "prob_png": [],
        "mask_png": [],
        "heatmap_png": [],
        "overlay_png": [],
        "chosen_index": best_idx,
        "scores": scores,
        "best_prob_array": prob_up[best_idx].cpu().numpy(),
        "best_hard_array": hard_up[best_idx].cpu().numpy()
    }

    output_prefix = Path(output_prefix)
    output_prefix.mkdir(parents=True, exist_ok=True)

    for i in indices:
        prob_i = prob_up[i].cpu().numpy()
        hard_i = hard_up[i].cpu().numpy()

        filename = Path(image_path).name
        stem = Path(image_path).stem
        # base = f"{output_prefix}_idx{i}"

        # 软概率 npy
        # prob_npy_path = base + ".prob.npy"
        # np.save(prob_npy_path, prob_i.astype(np.float32))
        # saved["prob_npy"].append(prob_npy_path)

        # 软概率灰度图
        prob_png_path = output_prefix / f"{stem}.png"
        imageio.imwrite(prob_png_path, (prob_i * 255).clip(0, 255).astype(np.uint8))
        saved["prob_png"].append(prob_png_path)

        # 硬掩码
        # mask_png_path = base + ".mask.png"
        # imageio.imwrite(mask_png_path, (hard_i * 255).astype(np.uint8))
        # saved["mask_png"].append(mask_png_path)

        # 概率热力图
        # heatmap_path = base + ".heatmap.png"
        # plt.figure(figsize=(4,4))
        # plt.imshow(prob_i, cmap=cmap, vmin=0, vmax=1)
        # plt.axis("off")
        # plt.colorbar(fraction=0.046, pad=0.04)
        # plt.tight_layout()
        # plt.savefig(heatmap_path, dpi=dpi)
        # plt.close()
        # saved["heatmap_png"].append(heatmap_path)

        # 软概率叠加：根据概率对 overlay_color 做线性混合
        # overlay = image.copy().astype(np.float32)
        # color_arr = np.array(overlay_color, dtype=np.float32)  # R,G,B
        # # 扩展到 (H,W,3) 做加权混合： overlay = (1 - alpha * prob) * img + (alpha * prob) * color
        # prob_3 = prob_i[..., None]
        # overlay = (1 - overlay_alpha * prob_3) * overlay + (overlay_alpha * prob_3) * color_arr
        # overlay = overlay.clip(0,255).astype(np.uint8)

        # overlay_path = base + ".prob_overlay.png"
        # plt.figure(figsize=(4,4))
        # plt.imshow(overlay)
        # plt.axis("off")
        # plt.tight_layout()
        # plt.savefig(overlay_path, dpi=dpi)
        # plt.close()
        # saved["overlay_png"].append(overlay_path)

        if verbose:
            # print(f"[save_soft_prob] idx={i} saved:")
            # print("  ", prob_npy_path)
            print("  ", prob_png_path)
            # print("  ", mask_png_path)
            # print("  ", heatmap_path)
            # print("  ", overlay_path)

    # if verbose:
    #     print(f"[save_soft_prob] best index = {best_idx}, score = {scores[best_idx]:.4f}")

    return saved

def segment(image_path, out_path, box, model_type, device="cuda"):
    # -------- 配置部分 --------
    if model_type == "tiny":
        ckpt_path = "/home/zr/paper_project/sam/checkpoints/sam2.1_hiera_tiny.pt"
        config_file = "../sam/sam2/configs/sam2.1/sam2.1_hiera_t.yaml"
    elif model_type == "small":
        ckpt_path = "/home/zr/paper_project/sam/checkpoints/sam2.1_hiera_small.pt"
        config_file = "../sam/sam2/configs/sam2.1/sam2.1_hiera_s.yaml"
    elif model_type == "base_plus":
        ckpt_path = "/home/zr/paper_project/sam/checkpoints/sam2.1_hiera_base_plus.pt"
        config_file = "/home/zr/paper_project/sam/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"
    elif model_type == "large":
        ckpt_path = "/home/zr/paper_project/sam/checkpoints/sam2.1_hiera_large.pt"
        config_file = "/home/zr/paper_project/sam/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"


    #是否归一化转像素
    box_normalized = False
    # inside_points_normalized = True
    # negative_points_normalized = True
    # use_polygon_points_as_positive = False


    # -------- 读取图像 --------
    image = load_image(image_path)
    H, W = image.shape[:2]

    box = denormalize_if_needed(box, W, H, box_normalized)
    # object_inside_points = denormalize_if_needed(ensure_points_format(object_inside_points), W, H,
    #                                              inside_points_normalized)
    # negative_points = denormalize_if_needed(ensure_points_format(negative_points), W, H, negative_points_normalized)

    # -------- 构建模型与预测器 --------
    sam2_model = build_sam2(config_file=config_file,ckpt_path=ckpt_path,device=device)
    # print("[DEBUG] Model built.")
    # print_some_params(sam2_model, k=3)

    predictor = SAM2ImagePredictor(
        sam2_model,
        mask_threshold=0.5,  # 或 0.6 / 0.7 逐步调
        max_hole_area=64,  # 根据分辨率调
        max_sprinkle_area=64
    )

    predictor.set_image(image)

    # -------- 生成 box (可选) --------
    if box is not None:
        if len(box) >= 2:
            box = np.array(box, dtype=np.float32)
        else:
            box = None
            print("[WARN] box 元素少于2个,无法生成 box")
        
        masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box,                   # 如果不想用 box 就写 box=None
        multimask_output=True      # 输出多个候选掩膜
        )

        # 选择得分最高的一个掩膜
        best_idx = int(np.argmax(scores))
        best_mask = masks[best_idx]

        # print("Candidate scores:", scores)
        # print("Chosen score:", scores[best_idx])

        # visualize_and_save(image, best_mask, save_path="./sam2_result.png")
        save_soft_prob(
            image=image,
            image_path=image_path,
            logits=logits,
            scores=scores,
            best_idx=best_idx,
            output_prefix=out_path,  # 会生成 sam2_result_idxX_*.*
            hard_threshold=0.5,
            save_all=False,      # 改 True 可以把所有候选都保存
            overlay_color=(0,255,0),
            overlay_alpha=0.55,
            verbose=True
        )

    else:
        mask = np.zeros((H, W), dtype=np.uint8)
        stem = Path(image_path).stem
        out_path = Path(out_path)
        mask_path = out_path / f"{stem}.png"
        imageio.imwrite(mask_path, mask)


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    segment(image_path="/home/zr/paper_project/dataset/openforensics_test/TP/0a05d2c1ec.png",out_path="/home/zr/paper_project/dataset/test/face",
            box=[128, 227, 508, 712],
            model_type="large", device="cuda")