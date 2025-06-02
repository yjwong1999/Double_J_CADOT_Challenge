import argparse
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
import ultralytics.data.build as build
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

class YOLOWeightedDataset(YOLODataset):
    def __init__(self, *args, mode="train", **kwargs):
        super(YOLOWeightedDataset, self).__init__(*args, **kwargs)

        self.train_mode = "train" in self.prefix
        self.count_instances()
        class_weights = np.sum(self.counts) / self.counts
        self.agg_func = np.mean

        self.class_weights = np.array(class_weights)
        self.weights = self.calculate_weights()
        self.probabilities = self.calculate_probabilities()

    def count_instances(self):
        self.counts = [0 for _ in range(len(self.data["names"]))]
        for label in self.labels:
            cls = label['cls'].reshape(-1).astype(int)
            for id in cls:
                self.counts[id] += 1
        self.counts = np.array(self.counts)
        self.counts = np.where(self.counts == 0, 1, self.counts)

    def calculate_weights(self):
        weights = []
        for label in self.labels:
            cls = label['cls'].reshape(-1).astype(int)
            if cls.size == 0:
                weights.append(1)
                continue
            weight = self.agg_func(self.class_weights[cls])
            weights.append(weight)
        return weights

    def calculate_probabilities(self):
        total_weight = sum(self.weights)
        return [w / total_weight for w in self.weights]

    def __getitem__(self, index):
        if not self.train_mode:
            return self.transforms(self.get_image_and_label(index))
        else:
            index = np.random.choice(len(self.labels), p=self.probabilities)
            return self.transforms(self.get_image_and_label(index))

# Monkey patch method
build.YOLODataset = YOLOWeightedDataset

def main(model_name, epochs, batch, imgsz):
    cwd = os.getcwd()
    parent_dir = os.path.dirname(cwd)
    yaml_file = f"{parent_dir}/mydata/data.yaml"

    model = YOLO(model_name)

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
        default=100,
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Input image size (default: 960)"
    )
    
    args = parser.parse_args()
    main(args.model_name, args.epoch, args.batch, args.imgsz)
