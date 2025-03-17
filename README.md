# Image Classification with ResNet  

🔬 **Patch-based Axonal Fiber Classification using ResNet50 for Whole-Brain Light Sheet Microscopy**  

## Overview  
This repository implements a **ResNet-based binary classification model** to determine whether an image patch contains **axonal fibers (Class 1)** or **not (Class 0)**. This model is part of a larger pipeline developed to **identify axonal fibers in whole-brain teravoxel-scale light sheet microscopy images**.  

## Task & Pipeline Integration  
- ✅ **Binary classification task**: Each image patch is labeled as:  
  - **Class 1** → Contains axonal fiber  
  - **Class 0** → No axonal fiber  
- ✅ **Model**: ResNet50 trained using PyTorch on microscopy image patches.  
- ✅ **Pipeline Integration**:  
  - The **inference script** generates a `metadata.json` file containing **patch names and classification outputs**.  
  - These predictions feed into the **next step of the pipeline** for whole-brain connectivity analysis.  

## Features  
- ✅ **ResNet-based classification from MONAI**: Supports pretrained and custom-trained models for patch-level classification.  
- ✅ **Efficient training pipeline**: Optimized for multi-GPU training and mixed precision.  
- ✅ **Inference with metrics**: Computes classification metrics such as precision, recall, and F1-score.  
- ✅ **Configurable setup**: Uses `config.yml` for flexible hyperparameter tuning.  
- ✅ **Reusable utilities**: Modularized codebase for easily integrating cosine scheduler into other projects.
- ✅ **Visualization Training and Metric Curves**: This repository includes a plotting utility to visualize training performance similar to wandb.

🚀 **Flexible Usage:** Although this model was developed for **axonal fiber classification**, it can be adapted for **other binary classification tasks** by providing a properly formatted dataset. The key requirement is a `data_dicts.pickle` file containing a **list of dictionaries**, where each dictionary includes:  
```python
{
    "image": "path/to/image1.tiff",
    "classification_label": 1  # or 0
}
```
---

## Training Performance  

I used a dataset of manually labeled patches: 
- 853  images as train dataset 
- 174  images as validation dataset 
- 177  images as test dataset

Each image has the size of 256x256x256 voxels

Below are the **training loss and evaluation metric curves**:  

**Hyperparameters:**
- downsample images to 64x64x64 voxels
- FB score (B=1) instead of F1 as the criterion to choose the best model as recall was more important than precision
- batch size of 64
- 5x more weights for class 1 in BCE loss function
   
📈 **Training Curves:**  
_(Attach `training_curves.png` here)_  

📊 **Performance on Test Set:**  
| Metric     | Precision | Recall | F1 Score |  
|------------|----------|--------|----------|  
||71.05|90.00|79.41|  


---
```bash
Repository Structure:

├── config.yml                                # Configuration file for training and inference  
├── train.py                                  # Script to train the ResNet model  
├── patch_classification_inference.py         # Inference script without metrics  
├── patch_classification_inference_with_metrics.py  # Inference script with evaluation metrics  
├── utils.py                                  # Utility functions for cosine scheduler
├── README.md                                 # Documentation  
├── requirements.txt                          # requirements packages  
```

## Installation  
Clone the repository and install dependencies:  
```bash
git clone https://github.com/yourusername/image-classification-resnet.git
cd image-classification-resnet
pip install -r requirements.txt
```

