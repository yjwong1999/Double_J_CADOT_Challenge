# IEEE ICIP 2025: CADOT Challenge

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/198dwtjhB3ETFRHRPLWNCi_bAr1g5213i?usp=sharing)

#### By [Yi Jie WONG](https://yjwong1999.github.io/) & [Jing Jie TAN](https://jingjietan.com/) et al

## TLDR
🤩 Our approach is simple, scale everything! We proposed a systematic `Tri-Axial Scaling` to approach Aerial Object Detection via:
1. Model Size
2. Dataset Size & Quality
3. Test-Time Inference

Basically, we achieve this `Tri-Axial Scaling` by:
1. Scaling model size
2. Diffusion Augmentation & Balanced Data Sampling
3. Test-Time Inference = Test-Time Augmentation + Ensemble Models

Basically, we notice that:
1. A larger model can learn more effectively from a noisy and imbalanced dataset compared to a smaller model.
2. A larger model benefits more from dataset size scaling.
3. A smaller model can also achieve performance comparable to a larger model through balanced data sampling.
4. A larger model tends to overfit when using a balanced data sampling strategy, but this can be mitigated by increasing the amount of data (hence, data scaling).

![Dataset Size Scaling](assets/Dataset_Size_Scaling.png)

## Instructions

❗Please refer our Colab link to try out our code seamlessly!

Conda environment
```bash
conda create --name yolo python=3.10.12 -y
conda activate yolo
```

Clone this repo
```bash
# clone this repo
git clone https://github.com/yjwong1999/Double_J_CADOT_Challenge.git
cd Double_J_CADOT_Challenge
```

Install dependencies
```bash
# Please adjust the torch version accordingly depending on your OS
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121

# Install Jupyter Notebook
pip install jupyter notebook==7.1.0

# install this version of ultralytics (for its dependencies)
pip install ultralytics==8.3.111

# uninstall default ultralytics and install my ultralytics that support Timm pretrained models
pip uninstall ultralytics -y
pip install git+https://github.com/DoubleY-BEGC2024/ultralytics-timm.git

# install this to use wighted box fusion
pip install ensemble-boxes

# Remaining dependencies
pip install pycocotools
pip install requests==2.32.3
pip install click==8.1.7
```

## Setup Dataset
```bash
# to convert the CADOT dataset from the default COCO format to YOLO annotation form
python setup_data.py
```

## Training Part 1 (without synthetic data)

❗Note that due to time constraints, we did not train all possible experiments. Hence, in general, our hyperparameters are chosen based on:
- If trained via balanced sampling, batch size = 8, image size = 960
- If trained without balanced sampling, batch size = 16, image size = 640
  
```bash
# train ResNext101-YOLO12 naively without tricks
python3 train.py --model-name "../cfg/yolo12-resnext101-timm.yaml" --epoch 100 --batch 16 --imgsz 640

# train yolo12n using balanced sampling
python3 train_balanced.py --model-name "yolo12n.pt" --epoch 100 --batch 8 --imgsz 960

# train yolo12s using balanced sampling
python3 train_balanced.py --model-name "yolo12s.pt" --epoch 50 --batch 8 --imgsz 960
```

## Training Part 2 (with synthetic data)
```bash
# setup our synthetic dataset
python setup_synthetic_data.py

# train yolo12x with synthetic data only
python3 train_balanced.py --model-name "yolo12x.pt" --epoch 100 --batch 16 --imgsz 640

# train yolo12x using balanced sampling and synthetic data
python3 train_balanced.py --model-name "yolo12x.pt" --epoch 100 --batch 8 --imgsz 960
```

## Inference
```bash
# Move all 5 models we trained into model directory
# Otherwise, download our trained models via the following scripts
bash download_our_model.sh

# Run the inference code
python3 infer.py --tta all
```
