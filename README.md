# EDNFC-Net
## PyTorch implementation of EDNFC-Net: Convolutional Neural Network with Nested Feature Concatenation for Nuclei-Instance Segmentation [[Publication Link]](https://ieeexplore.ieee.org/document/9053633)

![](https://github.com/shivgahlout/EDNFC-Net/blob/main/images/ednfc.gif)
### Requirements
- pytorch==1.7.1
- albumentations==1.2.1
- opencv-python==4.6.0
- torchmetrics==0.9.3
- numpy==1.20.2
- matplotlib==3.4.2
- scikit-image 3.4.2
- wandb (optional)
### Steps
1. Model is defined in `ednfc.py`
2. `main.py` --> training
3. `test.py` --> inference
### Sample predictions on test set of [ISIC 2017 dataset](https://challenge.isic-archive.com/data/#2017)
![](https://github.com/shivgahlout/EDNFC-Net/blob/main/images/predictions.gif)

##### Please cite this work as:
````
@INPROCEEDINGS{gehlot2020,
author={S. {Gehlot} and A. {Gupta} and R. {Gupta}},
booktitle={ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
title={EDNFC-Net: Convolutional Neural Network with Nested Feature Concatenation for Nuclei-Instance Segmentation},
year={2020},
pages={1389-1393},}
````

