AA-TransUNet: Attention Augmented TransUNet For Nowcasting Tasks.
---

[AA_TransUNet](https://github.com/YangYimin98/AA-TransUNet/blob/main/AA_TransUNet.png) Architecture.
---
![AA_TransUNet](https://user-images.githubusercontent.com/67627410/149968662-d3a732b3-b0b9-4285-84f4-a5e6995d7e8a.png)

Datasets & Pre-trained Models
---
If you are interesed in the dataset(precipitation maps & cloud cover dataset) used in this paper or the pre-trained AA_TransUNet models, please contact us:
    yimin.yang@student.maastrichtuniversity.nl or siamak.mehrkanoon@maastrichtuniversity.nl

Please put the dataset into "\dataset" directory for training and testing, and put the pre-trained models into "\results\Model_Saved" directory for future loading.

Usages
---

Required Dependencies:

    pip install -r requirements.txt

Training precipitation maps dataset:

    python train_precipitation.py
    
Training cloud cover dataset:

    python train_cloud_cover.py

After training or loading pre-trained models, you can evaluate model's performance by:

Evaluating precipitation maps dataset:

    python evaluate_precipitation.py
    
Evaluating cloud cover dataset:

    python evaluate_cloud_cover.py
 
For easier training, we also provide a colab demo:

   [Colab training demo for cloud cover dataset.](https://github.com/YangYimin98/AA-TransUNet/blob/main/AA_TransUNet_Demo.ipynb)

References
---

* [TransUNet](https://github.com/Beckschen/TransUNet)
* [self-attention-cv](https://github.com/The-AI-Summer/self-attention-cv)
* [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
* [SmaAt-UNet](https://github.com/HansBambel/SmaAt-UNet)

Authors
---

* Yimin Yang: yimin.yang@student.maastrichtuniversity.nl 
* Siamak Mehrkanoon: siamak.mehrkanoon@maastrichtuniversity.nl
