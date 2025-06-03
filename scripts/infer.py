import os
import json
import argparse
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
from tqdm import tqdm
from ensemble_boxes import weighted_boxes_fusion

# --- Helper Functions ---

def rotate_image(img, angle):
    if angle == 0:
        return img
    elif angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

def rotate_boxes(boxes, img_shape, angle):
    h, w = img_shape[:2]
    boxes_rot = boxes.copy()
    if angle == 0:
        return boxes_rot
    elif angle == 90:
        boxes_rot[:, [0, 2]] = boxes[:, [1, 3]]
        boxes_rot[:, [1, 3]] = w - boxes[:, [2, 0]]
    elif angle == 180:
        boxes_rot[:, [0, 2]] = w - boxes[:, [2, 0]]
        boxes_rot[:, [1, 3]] = h - boxes[:, [3, 1]]
    elif angle == 270:
        boxes_rot[:, [0, 2]] = h - boxes[:, [3, 1]]
        boxes_rot[:, [1, 3]] = boxes[:, [0, 2]]
    return boxes_rot

def adjust_brightness_contrast(img, alpha=1.2, beta=30):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def get_tta_variants(image, mode="all"):
    variants = []
    angles = [0, 90, 180, 270]

    if mode == "none":
        return [(image, 0)]
    
    if mode in ["rotate", "all"]:
        for angle in angles:
            img_rot = rotate_image(image, angle)
            variants.append((img_rot, angle))

    if mode in ["bc", "all"]:
        for angle in angles:
            img_rot = rotate_image(image, angle)
            img_bc = adjust_brightness_contrast(img_rot, alpha=1.2, beta=30)
            variants.append((img_bc, angle))

    return variants

def main():
    parser = argparse.ArgumentParser(description="YOLO Ensemble with TTA and WBF")
    parser.add_argument('--tta', type=str, default='all', choices=['none', 'rotate', 'bc', 'all'], help='TTA mode')
    args = parser.parse_args()

    # --- Model Ensemble Setup ---
    model_dir = os.path.abspath("../models")
    model_paths = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".pt")]

    models = [YOLO(path) for path in model_paths]

    test_images_dir = '../data/test'
    category_names = [
        'small-object', 'basketball field', 'building', 'crosswalk',
        'football field', 'graveyard', 'large vehicle', 'medium vehicle',
        'playground', 'roundabout', 'ship', 'small vehicle',
        'swimming pool', 'tennis court', 'train'
    ]
    categories = [{"id": idx, "name": name} for idx, name in enumerate(category_names)]
    images, annotations = [], []
    annotation_id = 1

    with open('../data/images_ids.json', 'r') as f:
        images_ids = json.load(f)

    for item in tqdm(images_ids["images"], desc="Processing images with Ensemble + TTA + WBF"):
        image_id = item["id"]
        file_name = item["file_name"]
        image_path = os.path.join(test_images_dir, file_name)

        image = cv2.imread(image_path)
        if image is None:
            continue
        h, w = image.shape[:2]
        images.append({"id": image_id, "width": w, "height": h, "file_name": file_name})

        boxes_list, scores_list, labels_list = [], [], []

        for img_aug, angle in get_tta_variants(image, mode=args.tta):
            for model in models:
                results = model(img_aug, conf=0.05, iou=0.7, verbose=False)[0]
                if results.boxes is None or len(results.boxes) == 0:
                    continue

                boxes = results.boxes.xyxy.cpu().numpy()
                scores = results.boxes.conf.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy().astype(int)

                boxes = rotate_boxes(boxes, img_aug.shape, angle)
                norm_boxes = boxes.copy()
                norm_boxes[:, [0, 2]] /= w
                norm_boxes[:, [1, 3]] /= h

                boxes_list.append(norm_boxes.tolist())
                scores_list.append(scores.tolist())
                labels_list.append(classes.tolist())

        if not boxes_list:
            continue

        boxes_fused, scores_fused, labels_fused = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            weights=None, iou_thr=0.7, skip_box_thr=0.05
        )

        for box, score, cls in zip(boxes_fused, scores_fused, labels_fused):
            x1, y1, x2, y2 = box
            x1 *= w; y1 *= h; x2 *= w; y2 *= h
            w_box, h_box = x2 - x1, y2 - y1

            annotations.append({
                "image_id": image_id,
                "category_id": int(cls),
                "bbox": [
                    round(float(x1), 2), round(float(y1), 2),
                    round(float(w_box), 2), round(float(h_box), 2)
                ],
                "score": round(float(score), 3)
            })
            annotation_id += 1

    coco_output = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open('predictions.json', 'w') as f:
        json.dump(coco_output['annotations'], f, indent=2)

    print(f"Saved {len(annotations)} ensemble+TTA+WBF annotations for {len(images)} images to predictions.json")

if __name__ == "__main__":
    main()
