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
        """
        Initialize the WeightedDataset.

        Args:
            class_weights (list or numpy array): A list or array of weights corresponding to each class.
        """

        super(YOLOWeightedDataset, self).__init__(*args, **kwargs)

        self.train_mode = "train" in self.prefix

        # You can also specify weights manually instead
        self.count_instances()
        class_weights = np.sum(self.counts) / self.counts
        self.agg_func = np.mean

        self.class_weights = np.array(class_weights)
        self.weights = self.calculate_weights()
        self.probabilities = self.calculate_probabilities()

    def count_instances(self):
        """
        Count the number of instances per class

        Returns:
            dict: A dict containing the counts for each class.
        """
        self.counts = [0 for i in range(len(self.data["names"]))]
        for label in self.labels:
            cls = label['cls'].reshape(-1).astype(int)
            for id in cls:
                self.counts[id] += 1

        self.counts = np.array(self.counts)
        self.counts = np.where(self.counts == 0, 1, self.counts)

    def calculate_weights(self):
        """
        Calculate the aggregated weight for each label based on class weights.

        Returns:
            list: A list of aggregated weights corresponding to each label.
        """
        weights = []
        for label in self.labels:
            cls = label['cls'].reshape(-1).astype(int)

            # Give a default weight to background class
            if cls.size == 0:
                weights.append(1)
                continue

            # Take mean of weights
            # You can change this weight aggregation function to aggregate weights differently
            # weight = np.mean(self.class_weights[cls])
            # weight = np.max(self.class_weights[cls])
            weight = self.agg_func(self.class_weights[cls])
            weights.append(weight)
        return weights

    def calculate_probabilities(self):
        """
        Calculate and store the sampling probabilities based on the weights.

        Returns:
            list: A list of sampling probabilities corresponding to each label.
        """
        total_weight = sum(self.weights)
        probabilities = [w / total_weight for w in self.weights]
        return probabilities

    def __getitem__(self, index):
        """
        Return transformed label information based on the sampled index.
        """
        # Don't use for validation
        if not self.train_mode:
            return self.transforms(self.get_image_and_label(index))
        else:
            index = np.random.choice(len(self.labels), p=self.probabilities)
            return self.transforms(self.get_image_and_label(index))

# Monkey patch method
build.YOLODataset = YOLOWeightedDataset

def main(model_name, epochs, batch_size, imgsz):
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
        batch=batch_size,
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
        project=f"runs/{os.path.splitext(os.path.basename(model_name))[0]}",           
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
        default=100,
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size for training (default: 16)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Image size for training (default: 960)"
    )

    args = parser.parse_args()

    main(args.model_name, args.epoch, args.batch, args.imgsz)
