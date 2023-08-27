# Adversarial Source Generation for Source-Free Domain Adaptation
This is the implementation of " Adversarial Source Generation for Source-Free Domain Adaptation ".
## Framework
![image](https://github.com/MFAaaaaaa/ASOGE/blob/master/frame/frame.png)
## Installation
* Clone this repository.
```bash
git clone https://github.com/MFAaaaaaa/ASOGE.git
```
## Datasets
* Please manually download the datasets [Office-31](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view), [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view), [VisDA-C](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification) from the official websites, and modify the path of images in ``` ./dataset ```
## Prerequisites

## Training
* Obtain pre-trained models on the Office-31 dataset.
```
python train_source --gpu 0 --data ./data/office-31/amazon --label_file ./data/amazon_9_1.pkl
```
* Source-Free Domain Adaptation on the Office-31 dataset.
```
python main --gpu 0 --source_path ./amazon_resnet_50_best.pkl --max_epoch 100  --data_path ./data/office-31/dslr --label_path ./data/dslr.pkl
```
