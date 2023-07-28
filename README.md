# Adversarial Source Generation for Source-Free Domain Adaptation
This is the implementation of " Adversarial Source Generation for Source-Free Domain Adaptation ".
## Framework
## Installation
* Clone this repository.
```bash
git clone --
```
## Datasets
* Please manually download the datasets [Office-31](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view), [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view), [VisDA-C](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification) from the official websites, and modify the path of images in ``` ./dataset ```
## Prerequisites

## Training
```
python train_source --gpu 0 --data ./data/office-31/amazon --label_file ./data/amazon_9_1.pkl

```
