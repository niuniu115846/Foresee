<p align="center">

  <h2 align="center">
  Foresee: Unlocking the Forgery Detection Potential of Vanilla MLLMs


  </h2>
  <p align="center">
    <a><strong>Rui Zuo</strong></a><sup>1</sup>
    ·
    <a><strong>Qinyue Tong</strong></a><sup>1</sup>
    ·
    <a href="https://scholar.google.com/citations?user=qx1yRVEAAAAJ&hl=zh-CN"><strong>Ziqian Lu</strong></a><sup>2</sup>
    ·
    <a><strong>Yunlong Yu</strong></a><sup>1</sup>
    <br>
    <a href="https://person.zju.edu.cn/lzmhome"><strong>Zhe-ming Lu</strong></a><sup>1</sup>
    <!-- <br> -->
    <br>
    <sup>1</sup>Zhejiang University, <sup>2</sup>Zhejiang Sci-Tech University
    <br>
    🧑‍💼 <b><i>Project Leader: Prof. Zhe-ming Lu</i></b>
    <br>
    <div align="center">
    <a href="https://arxiv.org/abs/2511.13442"><img src='https://img.shields.io/badge/arXiv-Foresee-red' alt='Paper PDF'></a>
    </div>
  </p>
</p>


![main_img](images/main.png)

## :mega: News
- **2025.11.17**: We’ve uploaded our paper *Unlocking the Forgery Detection Potential of Vanilla MLLMs:
A Novel Training-Free Pipeline* to arXiv! Welcome to **watch** 👀 this repository for the latest updates.


## 🤖 Foresee Overview

Foresee is a training-free MLLM-based pipeline tailored for interpretable image forgery detection and localization(IFDL). Foresee eliminates the need for additional training and enables a lightweight inference process, while surpassing existing MLLM-based methods in both tamper localization accuracy and the richness of textual explanations. Foresee augments vanilla MLLMs with a type-prior-driven reasoning process and supplies copy-move-specific feature extraction hints, enabling accurate identification of various manipulation types (e.g., splicing, copy-move) and providing more insightful textual explanations.


![teaser_img](images/teaser.png)

## 🚀 Installation

### Requirements

* Python == 3.10
* PyTorch == 2.5
* CUDA == 12.4

### Setup

### 1️⃣ Clone Foresee
```bash
git clone https://github.com/niuniu115846/Foresee.git
cd Foresee
pip install -r requirements.txt
```

### 2️⃣ Install Grounding DINO
Foresee relies on [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) for visual grounding.

Clone and install it in the same environment:

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
```

Download pre-trained model weights for GroundingDino.

```bash
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ../..
```

### 3️⃣ Install SAM 2

Foresee also depends on [SAM 2](https://github.com/facebookresearch/sam2) for segmentation.

Clone and install it in the same environment:

```bash
git clone https://github.com/facebookresearch/sam2.git && cd sam2
pip install -e .
```

Download pre-trained model weights for Sam2.

```bash
cd checkpoints && \
./download_ckpts.sh && \
cd ../..
```

### 4️⃣ Configure LLM API Keys

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your_openai_api_key_here"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

Or (Windows PowerShell):
```bash
setx OPENAI_API_KEY "your_openai_api_key_here"
setx OPENAI_BASE_URL "https://api.openai.com/v1"
```

### 5️⃣ Get Copy-Move hints

```bash
cd ffd
python run_point_method.py \
    --input_folder ./data/images \
    --output_folder ./results/visualizations \
    --mask_folder ./results/masks \
    --type surf
cd ..
```
The script allows customization through the following environment variables:

- `INPUT_FOLDER`: Path to the input image folder.
- `OUTPUT_FOLDER`: Path to save visualization results.
- `MASK_OUTPUT_FOLDER`: Path to save generated masks.
- `TYPE`: Feature detector type (sift / surf / akaze).


## 🎯 Test

You can test Foresee using the following script:

```bash
cd detection
python detection.py \
    --data_path dataset/CASIA 1.0/TP \
    --ffd_mask_path ffd/mask_output_path \
    --region_prompt_path ffd/hint_output_path \
    --out_path datasets/CASIA1.0_output_path \
    --doc_type TP \
    --llm_model gemini-2.5-pro-preview-03-25 \
    --use_ffd_mask
```

The script allows customization through the following environment variables:

- `DATA_PATH`: Path to the root directory of the dataset to be processed.
- `FFD_MASK_PATH`: Path to the precomputed FFD mask directory.
- `REGION_PROMPT_PATH`: Path to the region-level prompt results.
- `OUT_PATH`: Directory to save final segmentation and detection results.
- `DOC_TYPE`: Document type indicator. Options:
  - `TP` → Tampered  
  - `Au` → Authentic
- `LLM_MODEL`: Name of the LLM model used for region reasoning.
- `USE_FFD_MASK`: Boolean flag. If provided, enables FFD mask guidance for copy-move detection..

Modify these variables as needed to adapt the evaluation process to different datasets and setups.


## :clap: Acknowledgements
This project builds upon several outstanding research and engineering efforts:

- [GPT-5](https://openai.com/) developed by OpenAI  
- [Gemini 2.5 Pro](https://deepmind.google/technologies/gemini/) developed by Google DeepMind  
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) by IDEA Research  
- [SAM 2](https://ai.meta.com/sam2/) by Meta AI  

We sincerely thank the teams behind these models for their pioneering 
contributions to multimodal large models, visual grounding, and 
segmentation. Their open research and engineering efforts have 
significantly advanced the field and made this project possible.


## 🗂 Project Structure

```text
project_root/
├── datasets
│   ├── CASIA 1.0/           # CASIAv2 Tampered Images
│   │   ├── TP/
│   │   ├── Au/
│   │   ├── tp.list
│   │   └── au.list/
├── detection/
├── ffd/
├── llm/
├── sam/
├── GroundingDino/
└── README.md
```

## 📚 Citation

If you find this project useful, please consider citing:

```bibtex
@article{zuo2025unlocking,
  title={Unlocking the Forgery Detection Potential of Vanilla MLLMs: A Novel Training-Free Pipeline},
  author={Zuo, Rui and Tong, Qinyue and Lu, Zhe-Ming and Lu, Ziqian},
  journal={arXiv preprint arXiv:2511.13442},
  year={2025}
}
```


---

## 📬 Contact

* Author: Rui Zuo
* Email: [ruizuo@zju.edu.cn](ruizuo@zju.edu.cn)

---

⭐ If you find this repo helpful, feel free to star it!
