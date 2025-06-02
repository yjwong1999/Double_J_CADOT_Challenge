import argparse
from ultralytics import YOLO
import os

def main(model_name, epochs, imgsz, batch):
    # get current working directory (to make sure the code works anywhere in your device)
    cwd = os.getcwd()

    # get parent directory (because we are in scripts directory)
    parent_dir = os.path.dirname(cwd)

    # YAML file for dataset
    yaml_file = f"{parent_dir}/mydata/data.yaml"

    # Load model for transfer learning
    model = YOLO(model_name)

    # Train the model with custom settings
    results = model.train(
        data=yaml_file,
        batch=batch,
        epochs=epochs,
        imgsz=imgsz,
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
        required=True,
        help="Path to the model or model name (e.g., yolo12x.pt)"
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=300,
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (default: 640)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)"
    )
    
    args = parser.parse_args()

    main(args.model_name, args.epoch, args.imgsz, args.batch)
