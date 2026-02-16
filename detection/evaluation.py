import torch
import os
import torch.nn.functional as F
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from PIL import Image

def genertate_region_mask(masks ,batch_shape):
    """generate B 1 H W meaningful-region-mask for a batch of masks

    Args:
        batch_shape (_type_): _description_
    """
    meaningful_mask = torch.zeros_like(masks)
    for idx, shape in enumerate(batch_shape):
        meaningful_mask[idx, :, :shape[0], :shape[1]] = 1
    return meaningful_mask

def load_gt_binary(
    path: str,
    ignore_value: Optional[int] = None,
    *,
    positive_values: Optional[Tuple[int, ...]] = None,  # 多类别时，哪些值视为正类；None 表示 >0 为正
    threshold: Optional[float] = None                   # 灰度/概率型 GT 时，阈值化；None 表示 >0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取 GT，返回:
      gt:     (H,W) uint8 in {0,1}
      region: (H,W) float32 in {0,1}  有效区域掩码（1=评估，0=忽略）

    规则：
    - 如果提供 positive_values，则 gt = 1 当像素值 ∈ positive_values。
    - 否则如果提供 threshold，则 gt = 1 当像素值 >= threshold。
    - 否则默认 gt = 1 当像素值 > 0。
    - 如果提供 ignore_value，则 region = 0 当像素值 == ignore_value（其余为 1）。
    - 输入可为 .png/.jpg/.tif 等（读入为灰度）；或 .npy（二维数组）。
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
    else:
        # 以灰度方式读取（若原图是彩色标签图，会自动转灰度）
        img = Image.open(path).convert("L")
        arr = np.array(img)

    # region 基于原始值构建（在二值化之前）
    region = np.ones_like(arr, dtype=np.float32)
    if ignore_value is not None:
        region[arr == ignore_value] = 0.0

    # 生成二值 gt
    if arr.dtype == bool:
        gt_bool = arr
    elif positive_values is not None:
        gt_bool = np.isin(arr, positive_values)
    elif threshold is not None:
        gt_bool = arr.astype(np.float32) >= float(threshold)
    else:
        # 常见语义分割标签：>0 视为正类；0 视为背景
        gt_bool = arr.astype(np.float32) > 0.0

    # 对无效区域的 gt 可选清零（不影响后续，因为会用 region 过滤）
    gt = gt_bool.astype(np.uint8)

    return gt, region

def cal_confusion_matrix(predict, target, region_mask, threshold=0.5):
    """compute local confusion matrix for a batch of predict and target masks
    Args:
        predict (_type_): _description_
        target (_type_): _description_
        region (_type_): _description_
        
    Returns:
        TP, TN, FP, FN
    """
    predict = (predict > threshold).float()
    TP = torch.sum(predict * target * region_mask, dim=(1, 2, 3))
    TN = torch.sum((1-predict) * (1-target) * region_mask, dim=(1, 2, 3))
    FP = torch.sum(predict * (1-target) * region_mask, dim=(1, 2, 3))
    FN = torch.sum((1-predict) * target * region_mask, dim=(1, 2, 3))
    return TP, TN, FP, FN


def iou_from_counts(TP, FP, FN):
    return TP / (TP + FP + FN + 1e-8)

def resize_to(arr, size_hw, is_prob=True):
    """
    arr: (H,W)
    size_hw: (H_new, W_new)
    is_prob=True -> bilinear; else -> nearest
    """
    if arr.shape == size_hw:
        return arr
    t = torch.from_numpy(arr)[None, None, :, :].float()
    mode = "bilinear" if is_prob else "nearest"
    t = F.interpolate(t, size=size_hw, mode=mode, align_corners=False if mode=="bilinear" else None)
    return t.squeeze().numpy()

def load_mask(path, is_prob=False):
    """
    is_prob=False 时,用于GT(二值化).
    is_prob=True  时，用于预测概率图(若是0~255会归一化到0~1)。
    """
    # 读取为数组
    if path.endswith(".npy"):
        arr = np.load(path)
    else:
        img = Image.open(path).convert("L")
        arr = np.array(img)

    if is_prob:
        # 概率或强度图：若是 0~255，归一化到 0~1
        if arr.max() > 1:
            arr = arr / 255.0
        arr = arr.astype(np.float32)
    else:
        # GT：阈值二值化到 0/1
        if arr.max() > 1:
            arr = (arr > 127).astype(np.float32)
        else:
            arr = (arr > 0.5).astype(np.float32)

    return arr

def match_gt_path(gt_dir: str, pred_filename: str) -> Optional[str]:
    """
    GT 匹配规则：
    1) 优先用同名文件
    2) 尝试去
    掉常见预测后缀再拼 .png/.jpg
    """
    # direct = os.path.join(gt_dir, pred_filename)
    # if os.path.exists(direct):
    #     return direct

    # base = (
    #     pred_filename
    #     .replace(".prob.npy", "")
    #     .replace(".mask.png", "")
    #     .replace(".prob.png", "")
    # )
    # for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
    #     cand = os.path.join(gt_dir, base + suffix)
    #     if os.path.exists(cand):
    #         return cand
    # return None
    name, ext = os.path.splitext(pred_filename)
    # 1. 优先尝试加 _gt 后缀
    gt_name = f"{name}_gt{ext}"
    direct = os.path.join(gt_dir, gt_name)
    if os.path.exists(direct):
        return direct

    # 2. 尝试去掉常见预测后缀再加 _gt 后拼常见图片后缀
    base = (
        pred_filename
        .replace(".prob.npy", "")
        .replace(".mask.png", "")
        .replace(".prob.png", "")
    )
    for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
        cand = os.path.join(gt_dir, base + "_gt" + suffix)
        if os.path.exists(cand):
            return cand

    # 3. 最后尝试同名文件（极少情况）
    direct2 = os.path.join(gt_dir, pred_filename)
    if os.path.exists(direct2):
        return direct2

    return None

def _counts_to_metrics(tp: float, fp: float, fn: float) -> Tuple[float, float, float, float]:
    """
    从 TP/FP/FN 计算 (F1, P, R, IoU)，遇到除零时返回 0（可改为配置）。
    """
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return f1, p, r, iou

def collect_flat_pixels(
    pred_dir: str,
    gt_dir: str,
    resize_mode: str = "to_gt",   # "to_gt"|"to_pred"|"auto"
    ignore_value: Optional[int] = None,
    verbose: bool = True,
    # 可选：依赖注入（如你的函数名不同）
    load_mask: Optional[Callable[[str, bool], np.ndarray]] = None,
    load_gt_binary_fn: Optional[Callable[[str, Optional[int]], Tuple[np.ndarray, np.ndarray]]] = None,
    resize_to: Optional[Callable[[np.ndarray, Tuple[int, int], bool], np.ndarray]] = None,
) -> Dict:
    """
    读取 pred/gt，按需求对齐尺寸与忽略区域，返回按有效区域展平后的 per-image 向量列表。
    返回:
      {
        "per_image_probs": List[np.ndarray],   # 每张图的 prob_flat（有效区域内）
        "per_image_gts":   List[np.ndarray],   # 每张图的 gt_flat（有效区域内，0/1）
        "filenames_used":  List[str],
        "stats": {
          "num_images_used": int,
          "num_images_skipped_not_found": int,        # GT 未匹配
          "num_images_skipped_shape_mismatch": int,   # 在 resize_mode="none" 时形状不一致
          "num_images_skipped_empty_valid": int       # 有效区域全空
        }
      }
    """
    # 依赖默认
    if load_mask is None:
        # 假定项目里已有该函数
        load_mask = globals().get("load_mask")
        if load_mask is None:
            raise RuntimeError("load_mask 未提供。请传入 load_mask 或在全局定义。")
    if load_gt_binary_fn is None:
        load_gt_binary_fn = globals().get("load_gt_binary")
        if load_gt_binary_fn is None:
            raise RuntimeError("load_gt_binary 未提供。请传入 load_gt_binary_fn 或在全局定义。")
    if resize_to is None:
        resize_to = globals().get("resize_to")
        if resize_to is None:
            raise RuntimeError("resize_to 未提供。请传入 resize_to 或在全局定义。")

    pred_files = sorted(f for f in os.listdir(pred_dir) if not f.startswith('.'))
    if verbose:
        print(f"[INFO] {len(pred_files)} prediction files found.")

    per_image_probs: List[np.ndarray] = []
    per_image_gts:   List[np.ndarray] = []
    filenames_used:  List[str] = []

    skipped_not_found = 0
    skipped_shape_mismatch = 0
    skipped_empty_valid = 0
    used = 0

    for fn in pred_files:
        pred_path = os.path.join(pred_dir, fn)
        # name, ext = os.path.splitext(fn)
        # gt_path = os.path.join(gt_dir, f"{name}_gt{ext}")
        gt_path = match_gt_path(gt_dir, fn)
        if gt_path is None:
            skipped_not_found += 1
            if verbose:
                print(f"[WARN] GT not found for {fn}, skip.")
            continue

        # 读取
        prob = load_mask(pred_path, is_prob=True)  # (Hp, Wp) 浮点概率/分数
        gt, region = load_gt_binary_fn(gt_path, ignore_value=ignore_value)  # (Hg, Wg), (Hg, Wg)

        # 对齐尺寸
        if prob.shape != gt.shape:
            if resize_mode in ("to_gt", "auto"):
                prob = resize_to(prob, gt.shape, is_prob=True)
                # region 与 gt 同尺寸，无需变
            elif resize_mode == "to_pred":
                gt = resize_to(gt, prob.shape, is_prob=False)
                region = resize_to(region, prob.shape, is_prob=False)

        # 有效区域展平
        valid = region > 0.5
        if not np.any(valid):
            skipped_empty_valid += 1
            if verbose:
                print(f"[WARN] Empty valid region for {fn}, skip.")
            continue

        prob_flat = prob[valid].reshape(-1)
        gt_flat = gt[valid].astype(np.uint8).reshape(-1)

        per_image_probs.append(prob_flat)
        per_image_gts.append(gt_flat)
        filenames_used.append(fn)
        used += 1

        if verbose and used % 50 == 0:
            print(f"[INFO] collected {used} images...")

    if used == 0:
        raise RuntimeError("No images processed successfully in collect_flat_pixels().")

    stats = {
        "num_images_used": used,
        "num_images_skipped_not_found": skipped_not_found,
        "num_images_skipped_shape_mismatch": skipped_shape_mismatch,
        "num_images_skipped_empty_valid": skipped_empty_valid,
    }
    return {
        "per_image_probs": per_image_probs,
        "per_image_gts": per_image_gts,
        "filenames_used": filenames_used,
        "stats": stats,
    }

def compute_auc_from_flats(
    per_image_probs: Sequence[np.ndarray],
    per_image_gts: Sequence[np.ndarray],
    compute_pr_auc: bool = True,
) -> Dict:
    """
    基于已展平的 per-image 向量计算像素级 ROC-AUC / PR-AUC。
    返回:
      {
        "pixel_micro_roc_auc": float or None,
        "pixel_micro_pr_auc": float or None,
        "pixel_macro_roc_auc": float or None,
        "pixel_macro_pr_auc": float or None,
        "num_images_skipped_single_class": int
      }
    """
    assert len(per_image_probs) == len(per_image_gts)
    macro_aucs_roc: List[float] = []
    macro_aucs_pr:  List[float] = []
    skipped_single_class = 0

    # Macro：逐图
    for prob_flat, gt_flat in zip(per_image_probs, per_image_gts):
        u = np.unique(gt_flat)
        if len(u) < 2:
            skipped_single_class += 1
            continue
        try:
            macro_aucs_roc.append(roc_auc_score(gt_flat, prob_flat))
            if compute_pr_auc:
                macro_aucs_pr.append(average_precision_score(gt_flat, prob_flat))
        except ValueError:
            skipped_single_class += 1

    # Micro：合并
    all_probs = np.concatenate(per_image_probs, axis=0)
    all_gts   = np.concatenate(per_image_gts, axis=0)
    pixel_micro_roc_auc = None
    pixel_micro_pr_auc  = None
    if len(np.unique(all_gts)) >= 2:
        try:
            pixel_micro_roc_auc = float(roc_auc_score(all_gts, all_probs))
            if compute_pr_auc:
                pixel_micro_pr_auc = float(average_precision_score(all_gts, all_probs))
        except ValueError:
            pass

    pixel_macro_roc_auc = float(np.mean(macro_aucs_roc)) if len(macro_aucs_roc) > 0 else None
    pixel_macro_pr_auc  = float(np.mean(macro_aucs_pr))  if compute_pr_auc and len(macro_aucs_pr) > 0 else None

    return {
        "pixel_micro_roc_auc": pixel_micro_roc_auc,
        "pixel_micro_pr_auc": pixel_micro_pr_auc,
        "pixel_macro_roc_auc": pixel_macro_roc_auc,
        "pixel_macro_pr_auc": pixel_macro_pr_auc,
        "num_images_skipped_single_class": int(skipped_single_class),
    }

# def onepic_F1(pred_path, gt_path):
#     pred = load_mask(pred_path, is_prob=True)   # (H, W), 0~1 概率
#     gt   = load_mask(gt_path,   is_prob=False)  # (H, W), 0/1

#     # 2. 转成 (B, 1, H, W) 的 torch.Tensor
#     pred_t = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0)  # shape: (1,1,H,W)
#     gt_t   = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0)    # shape: (1,1,H,W)
#     # 全图有效
#     region_mask = torch.ones_like(gt_t)
#     TP, TN, FP, FN = cal_confusion_matrix(pred_t, gt_t, region_mask, threshold=0.5)
#     f1 = cal_F1(TP, TN, FP, FN)
#     return f1.item()

# def compute_f1_from_flats(
#     per_image_probs: Sequence[np.ndarray],
#     per_image_gts: Sequence[np.ndarray],
#     thresholds: Optional[Sequence[float]] = None,
#     auto_range: Tuple[float, float, int] = (0.05, 0.95, 19),
#     min_recall_constraint: Optional[float] = None,
# ) -> Dict:
#     """
#     基于 per-image 向量计算 F1/P/R/IoU 的 micro & macro 随阈值曲线，并选最优阈值。
#     返回结构与原 evaluate_F1_with_thresholds 类似：
#       {
#         "results_per_threshold": { t: {"micro": {...}, "macro": {...}} },
#         "best_micro": {"F1": float, "threshold": float},
#         "best_macro": {"F1": float, "threshold": float},
#         "thresholds": List[float],
#         "num_images": int
#       }
#     """
#     assert len(per_image_probs) == len(per_image_gts)
#     n_img = len(per_image_probs)

#     if thresholds is None:
#         start, end, num = auto_range
#         thresholds = list(np.linspace(float(start), float(end), int(num)))
#     else:
#         thresholds = [float(t) for t in thresholds]

#     # Micro: 每个阈值累计 TP/TN/FP/FN（TN 对 F1/IoU非必需，但保留）
#     micro_counts: Dict[float, Dict[str, float]] = {t: {"TP":0.0,"TN":0.0,"FP":0.0,"FN":0.0} for t in thresholds}
#     # Macro: 收集每张图的指标，再对图平均
#     macro_records: Dict[float, List[Dict[str, float]]] = {t: [] for t in thresholds}

#     # 逐图逐阈值
#     for prob_flat, gt_flat in zip(per_image_probs, per_image_gts):
#         gt_flat = gt_flat.astype(np.uint8)
#         for t in thresholds:
#             pred_bin = (prob_flat >= t).astype(np.uint8)

#             tp = float(np.sum((pred_bin == 1) & (gt_flat == 1)))
#             tn = float(np.sum((pred_bin == 0) & (gt_flat == 0)))
#             fp = float(np.sum((pred_bin == 1) & (gt_flat == 0)))
#             fn = float(np.sum((pred_bin == 0) & (gt_flat == 1)))

#             micro_counts[t]["TP"] += tp
#             micro_counts[t]["TN"] += tn
#             micro_counts[t]["FP"] += fp
#             micro_counts[t]["FN"] += fn

#             f1, p, r, iou = _counts_to_metrics(tp, fp, fn)
#             macro_records[t].append({"F1": f1, "P": p, "R": r, "IoU": iou})

#     # 汇总
#     results: Dict[float, Dict] = {}
#     best_micro = (-1.0, None)  # (F1, th)
#     best_macro = (-1.0, None)

#     for t in thresholds:
#         TP = micro_counts[t]["TP"]; TN = micro_counts[t]["TN"]
#         FP = micro_counts[t]["FP"]; FN = micro_counts[t]["FN"]

#         f1_micro, p_micro, r_micro, iou_micro = _counts_to_metrics(TP, FP, FN)

#         macro_list = macro_records[t]
#         F1_macro = float(np.mean([r["F1"] for r in macro_list])) if len(macro_list) else np.nan
#         P_macro  = float(np.mean([r["P"]  for r in macro_list])) if len(macro_list) else np.nan
#         R_macro  = float(np.mean([r["R"]  for r in macro_list])) if len(macro_list) else np.nan
#         IoU_macro= float(np.mean([r["IoU"] for r in macro_list])) if len(macro_list) else np.nan

#         results[t] = {
#             "micro": {
#                 "F1": f1_micro, "P": p_micro, "R": r_micro, "IoU": iou_micro,
#                 "TP": TP, "TN": TN, "FP": FP, "FN": FN
#             },
#             "macro": {
#                 "F1": F1_macro, "P": P_macro, "R": R_macro, "IoU": IoU_macro,
#                 "N_imgs": len(macro_list)
#             }
#         }

#         # 选最优阈值（可加召回约束）
#         if (min_recall_constraint is None) or (r_micro >= min_recall_constraint):
#             if f1_micro > best_micro[0]:
#                 best_micro = (f1_micro, t)
#         if (min_recall_constraint is None) or (R_macro >= min_recall_constraint):
#             if F1_macro > best_macro[0]:
#                 best_macro = (F1_macro, t)

#     return {
#         "results_per_threshold": results,
#         "best_micro": {"F1": best_micro[0], "threshold": best_micro[1]},
#         "best_macro": {"F1": best_macro[0], "threshold": best_macro[1]},
#         "thresholds": thresholds,
#         "num_images": n_img
#     }
def compute_f1_from_flats(
    per_image_probs: Sequence[np.ndarray],
    per_image_gts: Sequence[np.ndarray],
    thresholds: Optional[Sequence[float]] = None,
    auto_range: Tuple[float, float, int] = (0.05, 0.95, 19),
    min_recall_constraint: Optional[float] = None,
) -> Dict:
    """
    基于 per-image 向量计算 F1/IoU 的 micro & macro 随阈值曲线，并选最优阈值。
    仅在 results_per_threshold 中保留 F1 与 IoU（去除 P/R/TP/TN/FP/FN/N_imgs 等）。
    返回:
      {
        "results_per_threshold": { t: {"micro": {"F1","IoU"}, "macro": {"F1","IoU"}} },
        "best_micro": {"F1": float, "threshold": float},
        "best_macro": {"F1": float, "threshold": float},
        "thresholds": List[float],
        "num_images": int
      }
    """
    assert len(per_image_probs) == len(per_image_gts)
    n_img = len(per_image_probs)

    # 统一为 python float，避免 np.float64(...) 作为 key 的输出
    if thresholds is None:
        start, end, num = auto_range
        thresholds = [float(x) for x in np.linspace(float(start), float(end), int(num))]
    else:
        thresholds = [float(t) for t in thresholds]

    # Micro：累计 TP/FP/FN（F1/IoU 不需要 TN）
    micro_counts: Dict[float, Dict[str, float]] = {t: {"TP": 0.0, "FP": 0.0, "FN": 0.0} for t in thresholds}
    # Macro：收集每图的 (F1, IoU, R) 用于宏平均与召回约束（R 仅内部使用，不输出）
    macro_records: Dict[float, List[Tuple[float, float, float]]] = {t: [] for t in thresholds}

    # 逐图逐阈值
    for prob_flat, gt_flat in zip(per_image_probs, per_image_gts):
        gt_flat = gt_flat.astype(np.uint8)
        for t in thresholds:
            pred_bin = (prob_flat >= t).astype(np.uint8)

            tp = float(np.sum((pred_bin == 1) & (gt_flat == 1)))
            fp = float(np.sum((pred_bin == 1) & (gt_flat == 0)))
            fn = float(np.sum((pred_bin == 0) & (gt_flat == 1)))

            micro_counts[t]["TP"] += tp
            micro_counts[t]["FP"] += fp
            micro_counts[t]["FN"] += fn

            f1, p, r, iou = _counts_to_metrics(tp, fp, fn)  # p 未使用，仅 r 用于召回约束
            macro_records[t].append((f1, iou, r))

    # 汇总
    results: Dict[float, Dict] = {}
    best_micro = (-1.0, None)  # (F1, th)
    best_macro = (-1.0, None)

    for t in thresholds:
        TP = micro_counts[t]["TP"]; FP = micro_counts[t]["FP"]; FN = micro_counts[t]["FN"]
        f1_micro, p_micro, r_micro, iou_micro = _counts_to_metrics(TP, FP, FN)

        macro_list = macro_records[t]
        if len(macro_list) > 0:
            F1_macro = float(np.mean([f for (f, _, _) in macro_list]))
            IoU_macro = float(np.mean([i for (_, i, _) in macro_list]))
            R_macro = float(np.mean([r for (_, _, r) in macro_list]))
        else:
            F1_macro = np.nan
            IoU_macro = np.nan
            R_macro = -np.inf  # 确保在有召回约束时不会被选为最佳

        # 仅保留 F1 和 IoU
        results[t] = {
            "micro": {"F1": f1_micro, "IoU": iou_micro},
            "macro": {"F1": F1_macro, "IoU": IoU_macro}
        }

        # 选最优阈值（支持 micro/macro 的最小召回约束，仅内部使用，不输出 R）
        if (min_recall_constraint is None) or (r_micro >= min_recall_constraint):
            if f1_micro > best_micro[0]:
                best_micro = (f1_micro, t)
        if (min_recall_constraint is None) or (R_macro >= min_recall_constraint):
            if F1_macro > best_macro[0]:
                best_macro = (F1_macro, t)

    return {
        "results_per_threshold": results,
        "best_micro": {"F1": best_micro[0], "threshold": best_micro[1]},
        "best_macro": {"F1": best_macro[0], "threshold": best_macro[1]},
        "thresholds": thresholds,
        "num_images": n_img
    }

def evaluate_pixel_metrics(
    pred_dir: str,
    gt_dir: str,
    *,
    # 收集阶段
    resize_mode: str = "to_gt",       # "to_gt"|"to_pred"|"none"|"auto"
    ignore_value: Optional[int] = None,
    verbose: bool = True,
    # F1 阶段
    thresholds: Optional[Sequence[float]] = None,
    auto_range: Tuple[float, float, int] = (0.05, 0.95, 19),
    min_recall_constraint: Optional[float] = None,
    # AUC 阶段
    compute_pr_auc: bool = True
) -> Dict:
    """
    一次性完成：收集像素对 -> 计算 AUC (micro/macro) -> 计算 F1 曲线与最优阈值（micro/macro）。
    返回:
      {
        "auc": {...},
        "f1":  {...},   # 结构同 compute_f1_from_flats 返回
        "stats": {...}, # 收集阶段的统计
        "thresholds": [...]
      }
    """
    collected = collect_flat_pixels(
        pred_dir=pred_dir,
        gt_dir=gt_dir,
        resize_mode=resize_mode,
        ignore_value=ignore_value,
        verbose=verbose
    )
    per_image_probs = collected["per_image_probs"]
    per_image_gts   = collected["per_image_gts"]

    auc_block = compute_auc_from_flats(
        per_image_probs=per_image_probs,
        per_image_gts=per_image_gts,
        compute_pr_auc=compute_pr_auc
    )
    f1_block = compute_f1_from_flats(
        per_image_probs=per_image_probs,
        per_image_gts=per_image_gts,
        thresholds=thresholds,
        auto_range=auto_range,
        min_recall_constraint=min_recall_constraint
    )
    return {
        "auc": auc_block,
        "f1":  f1_block,
        "stats": collected["stats"],
        "thresholds": f1_block["thresholds"]
    }

if __name__ == "__main__":
    results = evaluate_pixel_metrics(
        pred_dir="/home/zr/paper_project/dataset/imd_result/vanilla_gemini/TP",
        gt_dir="/home/zr/paper_project/dataset/imd_test/Gt",
        resize_mode="to_gt",
        ignore_value=None,
        verbose=True,
        thresholds=None,
        auto_range=(0.05, 0.95, 19),
        min_recall_constraint=None,
        compute_pr_auc=True
    )
    import pprint
    pprint.pprint(results)