from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import warnings
from torchvision.ops import box_convert
import torch
import os
warnings.filterwarnings("ignore")

_MODEL = None  # lazy global cache

def get_groundingdino_box(
    image_path: str,
    text_prompt: str,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    *,
    device: str = "cuda",
    config_path: str = "../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    weights_path: str = "../GroundingDINO/weights/groundingdino_swint_ogc.pth",
    ensure_one: bool = True  # 若按阈值无结果，则放开阈值取top-1
) -> list | None:
    """
    返回该文本提示下置信度最高的一个框的像素坐标 [x1, y1, x2, y2]。
    若 ensure_one=False 且无候选，则返回 None。
    """

    if text_prompt == None or text_prompt == "None":
        return None
    
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model(config_path, weights_path, device=device)

    image_source, image = load_image(image_path)

    # 先按用户阈值筛选
    boxes, logits, phrases = predict(
        model=_MODEL,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    # 如无结果且需要保证至少一个，放开阈值后再取top-1
    if (logits is None or len(logits) == 0) and ensure_one:
        boxes, logits, phrases = predict(
            model=_MODEL,
            image=image,
            caption=text_prompt,
            box_threshold=0.0,
            text_threshold=text_threshold,
        )

    # 仍无结果
    if logits is None or len(logits) == 0:
        return None

    # 取 top-1
    top_idx = torch.argmax(logits).item()
    top_box = boxes[top_idx : top_idx + 1]  # [1,4], 归一化的 [cx,cy,w,h]

    # 转成像素级别 [x1,y1,x2,y2]
    H, W = image_source.shape[:2]
    xyxy_norm = box_convert(top_box, in_fmt="cxcywh", out_fmt="xyxy")
    xyxy_pix = (xyxy_norm * torch.tensor([W, H, W, H], dtype=xyxy_norm.dtype)).flatten()

    x1, y1, x2, y2 = [int(round(v)) for v in xyxy_pix.tolist()]
    # 边界裁剪
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W - 1, x2))
    y2 = max(0, min(H - 1, y2))

    return [x1, y1, x2, y2]

if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)
    print(current_file_path)
    model = load_model("../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "../GroundingDINO/weights/groundingdino_swint_ogc.pth", device="cuda")
    IMAGE_PATH = "/home/zr/paper_project/dataset/openforensics_test/TP/0a05d2c1ec.png"
    TEXT_PROMPT = "a male face in the middle of the screen"
    BOX_THRESHOLD = 0.35
    TEXT_THRESHOLD = 0.25

    image_source, image = load_image(IMAGE_PATH)
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=0.0,
        text_threshold=TEXT_THRESHOLD
    )
    # print("boxes:", boxes)
    # print("logits:", logits)
    # print("phrases:", phrases)
    # annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    # cv2.imwrite("annotated_image.jpg", annotated_frame)

    top_idx = torch.argmax(logits).item()                # 或者 values, indices = torch.topk(logits, k=1)
    boxes = boxes[top_idx:top_idx+1]                     # 保持形状 [1, 4]
    logits = logits[top_idx:top_idx+1]                   # 保持形状 [1]
    phrases = [phrases[top_idx]]                         # 对应的短语

    print("top1 box (normalized cx,cy,w,h):", boxes)
    print("top1 logit:", logits.item())
    print("top1 phrase:", phrases[0])

    # 如需像素坐标（将归一化的 [cx,cy,w,h] 转为 [x1,y1,x2,y2] 像素）
    H, W = image_source.shape[:2]
    xyxy_norm = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")          # 仍是归一化
    xyxy_pix = (xyxy_norm * torch.tensor([W, H, W, H], dtype=xyxy_norm.dtype)).round().to(torch.int)[0]
    x1, y1, x2, y2 = xyxy_pix.tolist()
    print(f"top1 box in pixels [x1,y1,x2,y2]: {x1}, {y1}, {x2}, {y2}")

    # 只用 top-1 做可视化
    annotated = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    cv2.imwrite("annotated_image_top1.jpg", annotated)
