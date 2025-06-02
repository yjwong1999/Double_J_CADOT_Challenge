# IEEE ICIP 2025: CADOT Challenge

#### By [Yi Jie WONG](https://yjwong1999.github.io/) & [Jing Jie TAN](https://jingjietan.com/) et al

## Instructions
Conda environment
```bash
conda create --name yolo python=3.10.12 -y
conda activate yolo
```

Clone this repo
```bash
# clone this repo
git clone https://github.com/jingjie00/cadot.git
cd cadot
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
python train.py --model-name "yolov5x.pt" --epoch 300
python train.py --model-name "yolo11x.pt" --epoch 300
python train.py --model-name "yolov8x.pt" --epoch 300
python train.py --model-name "yolov10x.pt" --epoch 300
python train.py --model-name "yolo12x.pt" --epoch 300
python train_cont.py --model-name "yolo12x.pt" --epoch 300
```
