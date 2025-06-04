## TLDR

❗Full Report Coming Soon!

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

<img src="../assets/Segmentation_Guided_Diffusion.jpg" alt="Diffusion Augmentation" width="800"> </br>
⬆️ Our diffusion augmentation pipeline converts annotations into synthetic image. </br>
This figure is adopted from my proposed method from another competition. </br>
I modified the pipeline to support bbox -> segmentation mask -> image generation. </br>
A more up to date figure will be updated here soon! </br>
To avoid overcomplicating this repo, we separate the code for diffusion augmentation in a separate [repo](https://github.com/yjwong1999/LOTR).

<img src="../assets/Dataset_Size_Scaling.png" alt="Dataset Size Scaling" width="800"> </br>
⬆️ Scaling Model Size vs Scaling Data Size vs Scaling Test-Time Inference </br>
Larger model is more effective in learning from imbalanced dataset. </br>
Larger model also benefits from data size scaling even in the presence of imbalanced class. </br>

<img src="../assets/Dataset_Balanced_Sampling.png" alt="Dataset Balanced Sampling" width="800"> </br>
⬆️ Scaling Model Size vs Scaling Data Quality vs Sacling Test-Time Inference </br>
Smaller model benefits more from balanced sampling as opposed to larger models. </br>
However, we see evidence of larger model (YOLO12s) to be better than smaller model (YOLO12n). </br>
We hyphothesized that bigger dataset is required to unlock full potential of YOLO12x. </br>

<img src="../assets/Test_Time_Scaling.png" alt="Test Time Scaling" width="800"> </br>
⬆️ Finally, we unleash the full potential of test-time scaling using ensemble model and TTA. </br>
We apply Test Time Augmentation to all models in our ensemble to increase the detection rate. </br>

[TODO](www.github.com) </br>
