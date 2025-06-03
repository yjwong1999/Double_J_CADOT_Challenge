## Our Trained Models

For this CADOT challenge, we leverage ensemble modeling to stack and combine the predictions of multiple YOLO models into a final prediction. 

Additionally, we apply test-time augmentation to enhance the detection performance of our final predictions. 

By combining these two techniques, we implement a form of test-time scaling, which is the central theme of this project.

Our trained models can be downloaded via `scripts/download_our_model.sh`. 

In total, we used 5 models in our prediction:
1. ResNext101-YOLO12 trained naively without tricks
2. YOLO12n trained with balanced sampling
3. YOLO12s trained with balanced sampling
4. YOLO12x trained with balanced sampling and augmentation (synthetic data)
5. YOLO12x trained with augmentation (synthetic data) only
