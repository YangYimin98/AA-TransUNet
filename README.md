AA-TransUNet: [Attention Augmented TransUNet For Nowcasting Tasks](https://arxiv.org/abs/2202.04996)
---

[AA_TransUNet](https://github.com/YangYimin98/AA-TransUNet/blob/main/model.eps) Architecture.
---
![AA_TransUNet](https://github.com/YangYimin98/AA-TransUNet/blob/main/model.png)

Datasets & Pre-trained Models
---
If you are interesed in the dataset(precipitation maps & cloud cover dataset) used in this paper，please visit [SmaAt-UNet](https://github.com/HansBambel/SmaAt-UNet) for further details.

For the pre-trained AA_TransUNet models, please contact us:
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

   [Colab training demo for cloud cover dataset.](https://github.com/YangYimin98/AA-TransUNet/blob/main/AA_TransUNet_Training_Demo_Cloud_Cover.ipynb)


Authors
---

* Yimin Yang: yimin.yang@student.maastrichtuniversity.nl 
* Siamak Mehrkanoon: siamak.mehrkanoon@maastrichtuniversity.nl

Citation
---
@article{yang2022aa,
  title={Aa-transunet: Attention augmented transunet for nowcasting tasks},
  author={Yang, Yimin and Mehrkanoon, Siamak},
  journal={arXiv preprint arXiv:2202.04996},
  year={2022}
}
