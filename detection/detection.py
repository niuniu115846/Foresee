import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from typing import Any, Iterable, Union
from PIL import Image
from pathlib import Path
import shutil
from llm.llm_predict import to_data_url, one_pic_llm_predict, two_pic_llm_predict
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_sam2 import segment,denormalize_if_needed
import ast
import re
from run_grounding_dino import get_groundingdino_box

def append_to_jsonl(
    name: str,
    description: str,
    llm_bbox: Any,
    groundingdino_bboxes: Any,
    jsonl_path: Union[str, Path],
) -> None:
    """

    参数:
      - meta: 包含 'description' 和 'boundingbox' 的字典
              例如: {"description": "xxx", "boundingbox": "10,20,30,40"}
      - rel: 通过读取列表文件每行得到的字符串，如 "images/a.jpg"
      - groundingdino_bboxes: 列表，原样写入到 'groundingdino_boundingbox' 字段
      - jsonl_path: 输出 JSON Lines 文件路径（不存在会自动创建目录并创建文件）

    """

    entry = {
        "name": name,
        "description": description,
        "boundingbox": llm_bbox,
        "groundingdino_boundingbox": groundingdino_bboxes,
    }

    jsonl_path = Path(jsonl_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False))
        f.write("\n")

def detection(data_path, ffd_mask_path = "/home/zr/paper_project/regionprompt_module/Copy_Move_Forgery_Detection/test/point/surf/CASIA1.0", region_prompt_path = "/home/zr/paper_project/regionprompt_module/Copy_Move_Forgery_Detection/result/surf/gpt-5/CASIA1.0_test/TP",out_path = "../dataset/CASIA1.0_result/TP", doc_type="TP", llm_model="gpt-5-chat",sam_model_type="large",use_box="groundingdino", use_ffd_mask=True):
    base_dir = Path(data_path)  # 数据集根目录
    region_prompt_dir = Path(region_prompt_path)
    # prompt = "Please determine whether this image has been tampered with. If it has been tampered with, generate a short phrase with location information to describe the tampered area, and provide the normalized coordinates of each tampered area's bounding box. The bounding box format is: [xmin, ymin, xmax, ymax]. Your output must be provided as a dictionary in the form {description: 'xxx', boundingbox: 'xxx'}. In description, you need to describe the tampered content; in boundingbox, you need to fill in the normalized coordinates of the tampered content. If the image has not been tampered with, both description and boundingbox should be None."

    if doc_type=="TP":
        list_path = 'tp_list.txt'
    elif doc_type=="Au":
        list_path = 'au_list.txt'

    list_path = Path(list_path)

    with open(base_dir / list_path, 'r', encoding='utf-8') as f:
        for line in f:
            rel = line.strip()
            if not rel:
                continue
            print("start rel:",rel)
            img = Image.open(base_dir / doc_type / rel)
            img_region_prompt_path = region_prompt_dir / rel
            H, W = img.height, img.width
            img_path = base_dir / doc_type / rel
            image_url = to_data_url(img_path)
            image_region_prompt_url = to_data_url(img_region_prompt_path)

            # 判断类型
            type_prompt = "This image has been tampered with. If this image is face image, tampering types of this image is 'face'. If not, the possible tampering types are:'copy-move': A region within the same image is selected, copied, and pasted elsewhere in the same image. This type often shows highly similar or repeated regions within the image, which may differ in size, angle, or position, but have very similar textures and content. 'splicing': Content from another image is cut and pasted into the current image, or a region from the current image is pasted into another image. This type typically appears as regions with textures, styles, or patterns that are noticeably different from the rest of the image.Please determine the tampering type and respond only with 'copy-move' or 'splicing'. Make sure your answer is correctly formatted.Special instructions:If you find multiple regions in the image that are highly similar (in texture or content), even if they differ in size, angle, or position, please give priority to 'copy-move'.Only choose 'splicing' when you clearly identify content that has been introduced from outside the original image (such as regions with distinctly different styles or textures).If you are unsure about the presence of external content, be more inclined to select 'copy-move'."
            type_prompt = "This is a tampered image. Please classify the type of tampering into one of the following four categories: face, copy-move, aigc, or others (including splicing, removal, and local enhancement).Please identify the tampering type and reply with only one of the following: “face”, “copy-move”, “others”, or “aigc”. Make sure your response format is correct.If the image is a facial image, the tampering type should be “face”.If not, please consider the following types: 'Copy-move': A region from the same image is copied and pasted elsewhere within the same image. This typically results in multiple highly similar or repeated regions, which may differ in size, angle, or position but have very similar texture and content. 'Splicing': Content from another image is cut and pasted into the current image, or a region from the current image is pasted into another image. This usually appears as areas with textures, styles, or patterns that are noticeably different from the rest of the image. 'Removal': A region of the image has been removed, possibly leaving behind artifacts. 'Local enhancement': Certain parts of the image appear excessively bright or dark. 'AIGC': AI-generated tampered images, which may show local confusion or inconsistencies. Special instructions: If you find multiple highly similar regions (in texture or content), even if they differ in size, angle, or position, prioritize 'copy-move'. Only select 'removal' or 'local enhancement' (and reply with 'others') when you clearly observe removal traces or inconsistent lighting/shadows. Select 'splicing' (and reply with 'others') only when you can clearly identify the introduction of external content (such as areas with distinctly different style or texture). If you are unsure whether external content is present, lean towards 'copy-move'. If you choose splicing, removal, or local enhancement, you must reply with 'others'. Finally, judge whether the image is forged by AIGC; if so, reply with 'aigc'."
            type_response = one_pic_llm_predict(image_url, type_prompt, model=llm_model)
            print(type_response)
            while type_response not in ["copy-move", "others", "face", "aigc","splicing"]:
                type_response = one_pic_llm_predict(image_url, type_prompt, model=llm_model)
                print(type_response)

            prompt_face = "You will be shown a human face image that is highly likely to have been manipulated by face editing or face swapping techniques (e.g., adding lipstick or glasses, skin smoothing/beautification, filters). Your task is to detect regions that are highly likely to be tampered, and please treat all regions mentioned in the detection requirements as tampered. Detection requirements: 1. Examine semantic anomalies, such as: dyed hair color on a middle-aged or elderly person, a child or a man wearing bright lipstick, or a woman or child with a prominent thick beard. 2. Inspect the eyeglass area carefully. Check for deformed frames and whether lens reflections/highlights are consistent with the rest of the face; look for abrupt changes in lighting/sharpness/noise. 3. Check whether the mouth region looks natural; watch for overly exaggerated smiles or unnatural exposed teeth. These are highly likely to be tampered regions. 4.Check the face for areas where the color is noticeably different from other regions, as these are highly likely to be tampered regions. Output requirements: Output only one dictionary in the following format: {'description':'xxx','boundingbox':'xxx'}, without any extra characters, punctuation, or spaces (no spaces between keys and values). The description must be a single noun, for example: 'beard', 'mustache', 'mouth', 'glasses', 'hair', 'eyebrows'. The bounding box must use normalized coordinates, formatted as '[x_min,y_min,x_max,y_max]' (as a string), where all four values are within the range [0,1]. If the image contains no human face or is not tampered, both description and boundingbox should be None. Output examples: Tampered example: {'description':'mouth','boundingbox':'[0.62,0.58,0.86,0.92]'} Untampered example: {'description':'None','boundingbox':'None'}"
            prompt_copymove = "You will be shown an image that is highly likely to have been tampered using the copy-move method. You need to detect possible tampering in this image. Detection requirements: 1. Make sure to carefully search for all possible repeated regions and give priority to identifying them as signs of tampering. 2.Focus on identifying repeated textures, patterns, or object instances in the image, especially those copied by affine transformations such as translation, rotation, scaling, or mirroring. If there are object instances with identical textures but different angles, sizes, or positions, those regions are highly likely to be tampered areas. 3.If you notice a region with jagged edges or sudden changes in sharpness or noise granularity, this is also a strong indicator of tampering.Output requirements: Output only one dictionary in the following format: {'description':'xxx','boundingbox':'xxx'}, without any extra characters, punctuation, or spaces (no spaces between keys and values).The description should be in the format of 'noun + prepositional phrase', and must contain only one prepositional phrase. For example: 'a yellow bird at the lower right corner of the image'. The bounding box must use normalized coordinates, formatted as '[x_min,y_min,x_max,y_max]' (as a string), where all four values are within the range [0,1].If the image is not tampered, both description and boundingbox should be None.Output examples: Tampered example: {'description':'a person sitting at the upper left corner of the rock surface','boundingbox':'[0.12,0.18,0.23,0.32]'} Untampered example: {'description':'None','boundingbox':'None'}"
            prompt_copymove_twopic = "You will be shown two images. The first image is highly likely to have been tampered using the copy-move method, and you need to detect possible tampering in it. The second image shows detection results for similar regions in the first image, but this detection method is not always effective. If no tampering is detected, the second image will be identical to the first one. In that case, rely on your own judgment. If the second image provides detection results, please prioritize the results shown in the second image. Detection requirements: 1. Make sure to carefully search for all possible repeated regions and give priority to identifying them as signs of tampering. 2. Focus on identifying repeated textures, patterns, or object instances in the image, especially those copied by affine transformations such as translation, rotation, scaling, or mirroring. If there are object instances with identical textures but different angles, sizes, or positions, those regions are highly likely to be tampered areas. 3. If you notice a region with jagged edges or sudden changes in sharpness or noise granularity, this is also a strong indicator of tampering. Output requirements: Output only one dictionary in the following format: {'description':'xxx','boundingbox':'xxx'}, without any extra characters, punctuation, or spaces (no spaces between keys and values).The description should be in the format of 'noun + prepositional phrase', and must contain only one prepositional phrase. For example: 'a yellow bird at the lower right corner of the image'.The bounding box must use normalized coordinates, formatted as '[x_min,y_min,x_max,y_max]' (as a string), where all four values are within the range [0,1]. If the image is not tampered, both description and boundingbox should be None. Output examples: Tampered example: {'description':''a green bird at the lower left corner of the image','boundingbox':'None'} Untampered example: {'description':'None','boundingbox':'None'}"
            prompt_others = "You will be shown an image that is highly likely to have been tampered with using splicing, removal, or local enhancement method. You need to detect possible tampering in this image.Detection requirements: 1.:If you notice a region with jagged edges or sudden changes in sharpness or noise granularity, this is a strong indicator of tampering by splicing or removal. 2.If there are areas in the image that are noticeably darker or brighter in contrast compared to other regions, this is likely the result of local enhancement tampering. 3.Make sure to carefully search for all regions that are noticeably different from the overall style, texture, color, or lighting of the image, as these are likely to be tampered areas. 3.Use semantic information in the image to determine if there are object instances that should not logically appear, such as a chicken in front of a predator, a normal-sized person in a distant landscape photo, or a flying bird photographed at close range in a place with human activity. Output requirements: Output only one dictionary in the following format: {'description':'xxx','boundingbox':'xxx'}, without any extra characters, punctuation, or spaces (no spaces between keys and values).The description should be in the format of 'noun + prepositional phrase', and must contain only one prepositional phrase. For example: 'a yellow bird at the lower right corner of the image'. The bounding box must use normalized coordinates, formatted as '[x_min,y_min,x_max,y_max]' (as a string), where all four values are within the range [0,1]. If the image is not tampered, both description and boundingbox should be None. Output examples: Tampered example: {'description':'a green bird at the lower left corner of the image','boundingbox':'[0.15,0.60,0.22,0.73]'} Untampered example: {'description':'None','boundingbox':'None'}"
            prompt_aigc = "You will be shown an image that is highly likely to have been tampered with using AIGC (AI-generated content) methods. You need to detect possible tampering in this image. Detection requirements: 1. If you notice areas in the image with unnatural textures, chaotic structures, blurry edges, inconsistent logic, or obvious signs of AI generation (such as unreasonable local content blending, abnormal object shapes, etc.), these are strong indicators of AIGC tampering. 2. Use semantic information in the image to determine if there are object instances or scenarios that should not logically appear, such as abnormal object boundaries, distorted details, or structural confusion. 3. Pay attention to possible AI-generated artifacts, such as repeated details, unnatural facial expressions, incorrect number of fingers, or physically impossible structures. Output requirements: Output only one dictionary in the following format: {'description':'xxx','boundingbox':'xxx'}, without any extra characters, punctuation, or spaces (no spaces between keys and values).The description should be in the format of 'noun + prepositional phrase', and must contain only one prepositional phrase. For example: 'a yellow bird at the lower right corner of the image'. The bounding box must use normalized coordinates, formatted as '[x_min,y_min,x_max,y_max]' (as a string), where all four values are within the range [0,1]. If the image is not tampered, both description and boundingbox should be None. Output examples: Tampered example: {'description':'a green bird at the lower left corner of the image','boundingbox':'[0.15,0.60,0.22,0.73]'} Untampered example: {'description':'None','boundingbox':'None'}"
            
            if type_response.strip().lower() == "copy-move":
                prompt = prompt_copymove_twopic
            elif type_response.strip().lower() == "others":
                prompt = prompt_others
            elif type_response.strip().lower() == "face":
                prompt = prompt_face
            elif type_response.strip().lower() == "aigc":
                prompt = prompt_aigc
            elif type_response.strip().lower() == "splicing":
                prompt = prompt_others
            else:
                raise ValueError(f"Unexpected tampering type: {type_response}")

            while True:
                if type_response.strip().lower() == "copy-move":
                    response = two_pic_llm_predict(image_url, image_region_prompt_url, prompt, model=llm_model)
                elif type_response.strip().lower() in ["others", "aigc", "face", "splicing"]:
                    response = one_pic_llm_predict(image_url, prompt, model=llm_model)
                else:
                    raise ValueError(f"Unexpected tampering type: {type_response}")
                
                print(response)
                print("rel:", rel)
                try:
                    response = ast.literal_eval(response)
                    break  # 解析成功，跳出循环
                except Exception:
                    # 解析失败则继续循环，重新跑 llm_predict
                    print("Failed to parse response, retrying...")
                    continue
            
            if use_ffd_mask and type_response.strip().lower() == "copy-move":
                append_to_jsonl(
                    name=rel,
                    description=response.get('description'),
                    llm_bbox=None,
                    groundingdino_bboxes=None,
                    jsonl_path=base_dir / f"{doc_type}_region_prompt_next_results_{llm_model}.jsonl",
                )
                rel_png = rel.rsplit('.', 1)[0] + '.png'
                rel_png = Path(rel_png)
                mask_path = Path(ffd_mask_path) / rel_png
                out_mask_path = Path(out_path) / rel_png
                shutil.copy2(mask_path, out_mask_path)
                print(f"Copied mask to {out_mask_path}")

            else:
                bbox_llm = ast.literal_eval(response['boundingbox']) if isinstance(response.get('boundingbox'), str) else response.get('boundingbox')
                description = response.get('description')
                print(type(description))
                response["boundingbox"] = denormalize_if_needed(bbox_llm, W, H, normalized=True)
                print(response)
                box_groundingdino = get_groundingdino_box(
                    image_path=img_path,
                    text_prompt=description,
                    box_threshold=0.35,
                    text_threshold=0.25,
                    device="cuda",
                    ensure_one=True
                )

                append_to_jsonl(
                    name=rel,
                    description=description,
                    llm_bbox=bbox_llm,
                    groundingdino_bboxes=box_groundingdino,
                    jsonl_path=base_dir / f"{doc_type}_region_prompt_next_results_{llm_model}.jsonl",
                )
                if use_box=="llm":
                    bbox_llm = denormalize_if_needed(bbox_llm, W, H, normalized=True)
                    bbox = bbox_llm
                elif use_box=="groundingdino":
                    bbox = box_groundingdino

                segment(image_path=img_path,out_path=out_path,
                box=bbox,
                model_type=sam_model_type, device="cuda")

            

if __name__ == "__main__":
    detection(data_path="/home/zr/paper_project/dataset/fakeclue",
              ffd_mask_path = "/home/zr/paper_project/regionprompt_module/Copy_Move_Forgery_Detection/test/point/surf/fakeclue",
              region_prompt_path = "/home/zr/paper_project/regionprompt_module/Copy_Move_Forgery_Detection/result/surf/gpt-5/fakeclue",
              out_path="/home/zr/paper_project/dataset/fakeclue_result/gemini/TP",
              doc_type="TP",
              llm_model="gemini-2.5-pro-preview-03-25",
              sam_model_type="large",
              use_box="groundingdino")
