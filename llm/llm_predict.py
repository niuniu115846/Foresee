from openai import OpenAI
import os
import requests
import base64
import argparse
import json
import jsonlines
import ast
import base64
from multiprocessing.pool import Pool
import time
from dashscope import MultiModalConversation

import os
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_BASE_URL"] = ""


def to_data_url(path):
    # 根据后缀设置 MIME
    ext = os.path.splitext(path)[1].lower()
    mime = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def one_pic_llm_predict(image_path, prompt, model="gpt-5-chat"):

    os.environ["OPENAI_API_KEY"] = ""
    os.environ["OPENAI_BASE_URL"] = ""
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{prompt}"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"{image_path}",
                        },
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content

def two_pic_llm_predict(image_path1, image_path2, prompt, model="gpt-5-chat"):
    import os
    from openai import OpenAI

    os.environ["OPENAI_API_KEY"] = ""
    os.environ["OPENAI_BASE_URL"] = ""
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{prompt}"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"{image_path1}",
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"{image_path2}",
                        },
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content

def ask_qwen_with_images(image_path1, image_path2, prompt, api_key="", model=""):
    """
    封装千问多模态对话API请求，支持2张图片和一个文本prompt输入。

    :param image_path1: 第一张图片路径
    :param image_path2: 第二张图片路径
    :param prompt: 用户输入的文本prompt
    :param api_key: 千问API Key
    :param model: 模型名，默认为"qwen-vl-plus"
    :return: 千问接口返回的文本内容
    """
    # 将图片读取为base64字符串
    def encode_image(img_path):
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    
    image1_b64 = encode_image(image_path1)
    image2_b64 = encode_image(image_path2)

    # 构造千问多模态消息体
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image1_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image2_b64}"}},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model,
        "messages": messages,
        "stream": False
    }

    try:
        response = requests.post(
            "https://api.probex.top/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=300
        )

        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            raise RuntimeError(f"请求失败，状态码: {response.status_code}\n错误信息: {response.text}")

    except requests.exceptions.Timeout:
        raise TimeoutError("请求超时：服务器没有响应")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"请求异常: {str(e)}")
import requests
import base64
import json

def ask_qwen_with_image(image_path, prompt, api_key="sk-ugdQjonfMKlIc1GtZrUWU8hmnGlfNC3cFUicw6PmyhfdGOvF", model="Qwen3-VL-235B-A22B-Instruct"):
    """
    封装千问多模态对话API请求，支持1张图片和一个文本prompt输入。

    :param image_path: 图片路径（本地）
    :param prompt: 用户输入的文本prompt
    :param api_key: 千问API Key
    :param model: 模型名，默认为"qwen-vl-plus"
    :return: 千问接口返回的文本内容
    """
    # 将图片读取为base64字符串
    def encode_image(img_path):
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    
    image_b64 = encode_image(image_path)

    # 构造千问多模态消息体
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model,
        "messages": messages,
        "stream": False
    }

    try:
        response = requests.post(
            "https://api.probex.top/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=300
        )

        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            raise RuntimeError(f"请求失败，状态码: {response.status_code}\n错误信息: {response.text}")

    except requests.exceptions.Timeout:
        raise TimeoutError("请求超时：服务器没有响应")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"请求发生错误: {e}")

if __name__ == "__main__":
    image_data_url = to_data_url("/home/zr/paper_project/dataset/CASIA1.0_test/TP/Sp_S_CNN_A_pla0060_pla0060_0192.jpg")
    image_data_url2 = to_data_url("/home/zr/paper_project/regionprompt_module/Copy_Move_Forgery_Detection/result/surf/gpt-5/CASIA1.0_test/TP/Sp_S_CNN_A_pla0060_pla0060_0192.jpg")
    prompt = "Please determine whether this image has been tampered with. If it has been tampered with, generate a short phrase with location information to describe the tampered area, and provide the normalized coordinates of each tampered area's bounding box. The bounding box format is: [xmin, ymin, xmax, ymax]. Your output must be provided as a dictionary in the form {description: 'xxx', boundingbox: 'xxx'}. In description, you need to describe the tampered content; in boundingbox, you need to fill in the normalized coordinates of the tampered content. If the image has not been tampered with, both description and boundingbox should be None."
    type_prompt = "This is a tampered image. The possible types of tampering are copy-move (where a region within the same image is selected, copied, and pasted elsewhere within the same image) or splicing (where a region from one image is cut and pasted onto a different image, or content from other images is copied into the current image). Please determine the type of tampering in this image and respond with only 'copy-move' or 'splicing'. Make sure your answer strictly follows the required format."
    type_prompt1 = "This is a tampered image. Please classify the type of tampering into one of the following four categories: face, copy-move, aigc, or others (including splicing, removal, and local enhancement).Please identify the tampering type and reply with only one of the following: “face”, “copy-move”, “others”, or “aigc”. Make sure your response format is correct.If the image is a facial image, the tampering type should be “face”.If not, please consider the following types: 'Copy-move': A region from the same image is copied and pasted elsewhere within the same image. This typically results in multiple highly similar or repeated regions, which may differ in size, angle, or position but have very similar texture and content. 'Splicing': Content from another image is cut and pasted into the current image, or a region from the current image is pasted into another image. This usually appears as areas with textures, styles, or patterns that are noticeably different from the rest of the image. 'Removal': A region of the image has been removed, possibly leaving behind artifacts. 'Local enhancement': Certain parts of the image appear excessively bright or dark. 'AIGC': AI-generated tampered images, which may show local confusion or inconsistencies. Special instructions: If you find multiple highly similar regions (in texture or content), even if they differ in size, angle, or position, prioritize 'copy-move'. Only select 'removal' or 'local enhancement' (and reply with 'others') when you clearly observe removal traces or inconsistent lighting/shadows. Select 'splicing' (and reply with 'others') only when you can clearly identify the introduction of external content (such as areas with distinctly different style or texture). If you are unsure whether external content is present, lean towards 'copy-move'. Finally, judge whether the image is forged by AIGC; if so, reply with 'aigc'."
    prompt_copymove = "You will be shown an image that is highly likely to have been tampered using the copy-move method. You need to detect possible tampering in this image. Detection requirements:1. Make sure to carefully search for all possible repeated regions and give priority to identifying them as signs of tampering. 2.Focus on identifying repeated textures, patterns, or object instances in the image, especially those copied by affine transformations such as translation, rotation, scaling, or mirroring. If there are object instances with identical textures but different angles, sizes, or positions, those regions are highly likely to be tampered areas. 3.If you notice a region with jagged edges or sudden changes in sharpness or noise granularity, this is also a strong indicator of tampering.Output requirements: Output only one dictionary in the following format: {'description':'xxx','boundingbox':'xxx'}, without any extra characters, punctuation, or spaces (no spaces between keys and values).The description should be in the format of 'noun + prepositional phrase', and must contain only one prepositional phrase. For example: 'a yellow bird at the lower right corner of the image'. The bounding box must use normalized coordinates, formatted as '[x_min,y_min,x_max,y_max]' (as a string), where all four values are within the range [0,1].If the image is not tampered, both description and boundingbox should be None.Output examples: Tampered example: {'description':'a person sitting at the upper left corner of the rock surface','boundingbox':'[0.12,0.18,0.23,0.32]'} Untampered example: {'description':'None','boundingbox':'None'}"
    prompt_splicing = "You will be shown an image that is highly likely to have been tampered with using the splicing method. You need to detect possible tampering in this image.Detection requirements: 1.:If you notice a region with jagged edges or sudden changes in sharpness or noise granularity, this is a strong indicator of tampering. 2.Make sure to carefully search for all regions that are noticeably different from the overall style, texture, color, or lighting of the image, as these are likely to be tampered areas. 3.Use semantic information in the image to determine if there are object instances that should not logically appear, such as a chicken in front of a predator, a normal-sized person in a distant landscape photo, or a flying bird photographed at close range in a place with human activity. Output requirements: Output only one dictionary in the following format: {'description':'xxx','boundingbox':'xxx'}, without any extra characters, punctuation, or spaces (no spaces between keys and values).The description should be in the format of 'noun + prepositional phrase', and must contain only one prepositional phrase. For example: 'a yellow bird at the lower right corner of the image'. The bounding box must use normalized coordinates, formatted as '[x_min,y_min,x_max,y_max]' (as a string), where all four values are within the range [0,1]. If the image is not tampered, both description and boundingbox should be None. Output examples: Tampered example: {'description':'a green bird at the lower left corner of the image','boundingbox':'[0.15,0.60,0.22,0.73]'} Untampered example: {'description':'None','boundingbox':'None'}"
    prompt_copymove_twopic = "You will be shown two images. The first image is highly likely to have been tampered using the copy-move method, and you need to detect possible tampering in it. The second image shows detection results for similar regions in the first image, but this detection method is not always effective. If no tampering is detected, the second image will be identical to the first one.If the second image provides detection results, please prioritize the results shown in the second image. Detection requirements: 1. Make sure to carefully search for all possible repeated regions and give priority to identifying them as signs of tampering. 2. Focus on identifying repeated textures, patterns, or object instances in the image, especially those copied by affine transformations such as translation, rotation, scaling, or mirroring. If there are object instances with identical textures but different angles, sizes, or positions, those regions are highly likely to be tampered areas. 3. If you notice a region with jagged edges or sudden changes in sharpness or noise granularity, this is also a strong indicator of tampering. Output requirements: Output only one dictionary in the following format: {'description':'xxx','boundingbox':'xxx'}, without any extra characters, punctuation, or spaces (no spaces between keys and values).The description should be in the format of 'noun + prepositional phrase', and must contain only one prepositional phrase. For example: 'a yellow bird at the lower right corner of the image'.The bounding box must use normalized coordinates, formatted as '[x_min,y_min,x_max,y_max]' (as a string), where all four values are within the range [0,1]. If the image is not tampered, both description and boundingbox should be None.Output examples: Tampered example: {'description':'a person sitting at the upper left corner of the rock surface','boundingbox':'[0.12,0.18,0.23,0.32]'} Untampered example: {'description':'None','boundingbox':'None'}"
    prompt_copymove_twopic = "You will be shown two images. The first image is highly likely to have been tampered using the copy-move method, and you need to detect possible tampering in it. The second image shows detection results for similar regions in the first image, but this detection method is not always effective. If no tampering is detected, the second image will be identical to the first one. In that case, rely on your own judgment. If the second image provides detection results, please prioritize the results shown in the second image. In that case, your output must be: 'use detection result'. Detection requirements: 1. Make sure to carefully search for all possible repeated regions and give priority to identifying them as signs of tampering. 2. Focus on identifying repeated textures, patterns, or object instances in the image, especially those copied by affine transformations such as translation, rotation, scaling, or mirroring. If there are object instances with identical textures but different angles, sizes, or positions, those regions are highly likely to be tampered areas. 3. If you notice a region with jagged edges or sudden changes in sharpness or noise granularity, this is also a strong indicator of tampering. Output requirements: If the second image does not provide the detection result of the tampered area in the first image, output only one dictionary in the following format: {'description':'xxx','boundingbox':'xxx'}, without any extra characters, punctuation, or spaces (no spaces between keys and values).The description should be in the format of 'noun + prepositional phrase', and must contain only one prepositional phrase. For example: 'a yellow bird at the lower right corner of the image'.The bounding box must use normalized coordinates, formatted as '[x_min,y_min,x_max,y_max]' (as a string), where all four values are within the range [0,1]. If the image is not tampered, both description and boundingbox should be None. Output examples: Tampered example: 'use detection result' or {'description':'a person sitting at the upper left corner of the rock surface','boundingbox':'[0.12,0.18,0.23,0.32]'} Untampered example: {'description':'None','boundingbox':'None'}"
    # response = one_pic_llm_predict(image_data_url, type_prompt1, model="gpt-5-chat")
    # response = ask_qwen_with_images("/home/zr/paper_project/dataset/columbia_test/TP/canong3_canonxt_sub_03.png","/home/zr/paper_project/regionprompt_module/Copy_Move_Forgery_Detection/result/sift/gpt-5/columbia_test/TP/canong3_canonxt_sub_03.png", prompt_copymove_twopic, model="Qwen3-VL-235B-A22B-Instruct", api_key="sk-ugdQjonfMKlIc1GtZrUWU8hmnGlfNC3cFUicw6PmyhfdGOvF")
    #response = ask_qwen_with_image("/home/zr/paper_project/dataset/columbia_test/TP/canong3_canonxt_sub_03.png", type_prompt, model="Qwen3-VL-235B-A22B-Instruct", api_key="sk-ugdQjonfMKlIc1GtZrUWU8hmnGlfNC3cFUicw6PmyhfdGOvF")
    # print(response)
    response = two_pic_llm_predict(image_data_url, image_data_url2, prompt_copymove_twopic, model="gpt-5-chat")
    print(response)
    print(type(response))