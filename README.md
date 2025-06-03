# IEEE ICIP 2025: CADOT Challenge

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/198dwtjhB3ETFRHRPLWNCi_bAr1g5213i?usp=sharing)

#### By [Yi Jie WONG](https://yjwong1999.github.io/) & [Jing Jie TAN](https://jingjietan.com/) et al

## Instructions

Please refer our Colab link to try out our code seamlessly!

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

# Remaining dependencies (for detection and etc)
pip install ultralytics==8.1
pip install pycocotools
pip install requests==2.32.3
pip install click==8.1.7
pip install opendatasets==0.1.22
```

## Data structure
Since we use YOLO as our detection model, we have to organize our dataset following the YOLO format. The `setup_data.py` code will automatically take the raw data from CADOT and convert it into YOLO format. The `mydata` directory will store the training data for our YOLO model.
```
cadot/mydata
├── images
│   └── train  
│   └── val  
├── labels
│   └── train  
│   └── val   
```

## Instructions
After installing all the dependencies, run the following codes:
```bash
# setup the dataset into YOLO format
python setup_data.py

# train the model
python train.py --model-name "yolo12x.pt" --epoch 100

# train with balanced sampling
python train_balanced.py --model-name "yolo12n.pt" --epoch 100
```
