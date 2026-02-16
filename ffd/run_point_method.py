import cv2
import os
import numpy as np
from Detector.SiftDetector import SiftDetector
from Detector.AkazeDetector import AkazeDetector
from Detector.SurfDetector import SurfDetector
import argparse


def get_rectangles_from_drawrectangle(keypoints1, keypoints2):
    # 只处理 cRectangle == 0 的情况，和 DrawRectangle 画法一致
    k1x = np.max(keypoints1, axis=0)
    k1n = np.min(keypoints1, axis=0)
    k2x = np.max(keypoints2, axis=0)
    k2n = np.min(keypoints2, axis=0)
    # 框1
    rect1 = (int(k1n[0]) - 10, int(k1n[1]) - 10, int(k1x[0]) + 10, int(k1x[1]) + 10)
    # 框2
    rect2 = (int(k2n[0]) - 10, int(k2n[1]) - 10, int(k2x[0]) + 10, int(k2x[1]) + 10)
    return [rect1, rect2]

def save_mask_from_rectangles(img_shape, rectangles, mask_path):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    for rect in rectangles:
        x1, y1, x2, y2 = rect
        # 防止越界
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, img_shape[1]), min(y2, img_shape[0])
        mask[y1:y2, x1:x2] = 255
    cv2.imwrite(mask_path, mask)

def copymove_region_detect(input_folder, output_folder, mask_folder, type="sift"):
    """
    对 input_folder 文件夹下所有图片做 SIFT/SURF/AKAZE 伪造检测，结果按原文件名存到 output_folder。
    匹配点不足或报错时直接保存原图。
    """
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(mask_folder, exist_ok=True)
    img_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

    for filename in os.listdir(input_folder):
        if any(filename.lower().endswith(ext) for ext in img_exts):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)

            if img is None:
                print(f"无法读取图片: {img_path}")
                continue
            
            detector = None
            result_img = None
            mask_path = os.path.join(mask_folder, filename)
            rectangles = []

            try:
                if type == "surf":
                    detector = SurfDetector(img)
                elif type == "akaze":
                    detector = AkazeDetector(img)
                elif type == "sift":
                    detector = SiftDetector(img)
                else:
                    raise ValueError("type must be 'surf', 'akaze' or 'sift'")

                result_img = detector.image

                draw = getattr(detector, 'Draw', None)
                if (draw is not None 
                    and hasattr(draw, 'keypoints1') and hasattr(draw, 'keypoints2')
                    and len(draw.keypoints1) > 0 and len(draw.keypoints2) > 0):
                    rectangles = get_rectangles_from_drawrectangle(draw.keypoints1, draw.keypoints2)
                    save_mask_from_rectangles(img.shape, rectangles, mask_path)
                    print(f"已保存mask: {mask_path}")
                else:
                    cv2.imwrite(mask_path, np.zeros(img.shape[:2], dtype=np.uint8))
                    print(f"没有匹配到任何关键点，已保存全黑mask: {mask_path}")

            except Exception as e:
                print(f"{filename} 检测失败（{e}），保存原图和全黑mask")
                result_img = img  # 检测失败时，直接用原图
                cv2.imwrite(mask_path, np.zeros(img.shape[:2], dtype=np.uint8))

            result_path = os.path.join(output_folder, filename)
            cv2.imwrite(result_path, result_img)
            print(f"已保存: {result_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copy-move detection of point-based method using SIFT / SURF / AKAZE"
    )

    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Path to the input image folder"
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Path to save visualization results"
    )

    parser.add_argument(
        "--mask_output_folder",
        type=str,
        required=True,
        help="Path to save generated masks"
    )

    parser.add_argument(
        "--type",
        type=str,
        default="surf",
        choices=["sift", "surf", "akaze"],
        help="Feature detector type (sift / surf / akaze)"
    )

    args = parser.parse_args()

    copymove_region_detect(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        mask_output_folder=args.mask_output_folder,
        type=args.type
    )


