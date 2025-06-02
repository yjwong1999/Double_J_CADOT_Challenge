import argparse
from ultralytics import YOLO
import os

def main(model_name, epochs):
    # get current working directory (to make sure the code works anywhere in your device)
    cwd = os.getcwd()

    # get parent directory (because we are in scripts direcotry)
    parent_dir = os.path.dirname(cwd)  

    # YAML file for dataset
    yaml_file = f"{parent_dir}/mydata/data.yaml"

    # Load model for transfer learning
    model = YOLO(model_name)

    # Train the model with custom settings
    results = model.train(
        data=yaml_file,
        batch=16,
        epochs=epochs,
        imgsz=640,
        plots=True,
        flipud=0.5,
        mixup=0.2,
        close_mosaic=0,
        optimizer="SGD",      
        momentum=0.9,
        weight_decay=0.0005,
        lr0=0.01,
        lrf=0.7,
        val=False,   
        project=f"runs/{model_name[:-3]}",             
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO with specified model and epochs.")
    parser.add_argument(
        "--model-name",
        type=str,
        help="Path to the model or model name (default: yolov5xu.pt)"
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=300,
        help="Number of training epochs (default: 100)"
    )
    
    args = parser.parse_args()

    main(args.model_name, args.epoch)
