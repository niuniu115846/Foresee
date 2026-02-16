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
    <a><strong>Sixue Men</strong></a><sup>1</sup>
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
python run_point_method.py
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
python detection.py
```

The script allows customization through the following environment variables:

- `WEIGHT_PATH`: Path to the directory containing the FakeShield model weights.

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
## 📦 Usage

### Quick Start

```bash
python demo.py --input example.jpg
```

### Detailed Usage

Explain important arguments or configs.

```bash
python main.py \
  --config configs/default.yaml \
  --input data/example
```

---

## 📊 Experiments & Results

Describe datasets, evaluation metrics, and main results.

* Dataset(s): XXX
* Metrics: Accuracy / F1 / AUC / mIoU / etc.

(Optional: add a result table or figure)

---

## 🧪 Reproducibility

Provide instructions to reproduce the main results.

```bash
bash scripts/run_experiment.sh
```

Mention random seeds, checkpoints, or hardware if necessary.

---

## 🗂 Project Structure

```text
project_root/
├── data/
├── configs/
├── models/
├── scripts/
├── utils/
├── main.py
└── README.md
```

---

## 📎 Pretrained Models (Optional)

| Model   | Dataset | Link          |
| ------- | ------- | ------------- |
| Model‑A | XXX     | [Download](#) |

---

## 📚 Citation

If you find this project useful, please consider citing:

```bibtex
@article{yourpaper2025,
  title={Title of Your Paper},
  author={Author1 and Author2},
  journal={Conference / Journal},
  year={2025}
}
```

---

## 🤝 Acknowledgements

Mention related projects, codebases, or collaborators you built upon.

---

## 📄 License

This project is released under the **MIT License** (or Apache 2.0 / GPL / etc.). See `LICENSE` for details.

---

## 📬 Contact

* Author: Your Name
* Email: [your@email.com](mailto:your@email.com)
* Homepage: [https://yourpage.com](https://yourpage.com)

---

⭐ If you find this repo helpful, feel free to star it!
