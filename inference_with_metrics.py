"""
This code is written by Ahmadreza Attarpour (a.attarpour@mail.utoronto.ca)
It evaluate a classification model trained on 3D patches of microscopy images.

main inputs:
    - model_path: path of the trained model
    - config: path of the config file
    - data_dicts: path of the data_dicts file
    - binarization_threshold: threshold to binarize the model; between 0-1

outputs:
    - print the image path, label, and prediction for each image in the test dataset
    - print the average precision, recall, and F1 score for the test dataset

"""

# -------------------------------------------------------
# load libraries
# -------------------------------------------------------
import os
import pickle
import numpy as np
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose,
    ScaleIntensityd,
    EnsureTyped,
    AddChanneld,
    Transform,
    Resized    
)
import torch
import torch.nn as nn
import argparse
import yaml
import tifffile
from monai.networks.nets import ResNet
from sklearn.metrics import precision_score, recall_score, f1_score

# -------------------------------------------------------
# Define transforms for image and segmentation
# ------------------------------------------------------- 
def train_val_transforms(cfg):
    # Get configuration parameters
    temp_value= cfg['general'].get('main_input_img_size', 256)
    main_input_img_size = (temp_value, temp_value, temp_value) # the main input to the model

    temp_value= cfg['general'].get('downsample_size', 128)
    downsample_size = (temp_value, temp_value, temp_value) # the main input to the model
    
    # image loader
    class MyLoadImage(Transform):
        def __call__(self, image_dict):
            # load the .tiff files they are HxWxD
            if isinstance(image_dict['image'], str):
                image_dict['img_path'] = image_dict['image']
                image_dict['image'] = tifffile.imread(image_dict['image'])

                # change the order axis of the image from DHW to HWD
                image_dict['image'] = np.moveaxis(image_dict['image'], 0, 2)
                image_dict['image'] = np.moveaxis(image_dict['image'], 0, 1)

            return image_dict

    # Validation transforms
    transforms = Compose(
        [
            MyLoadImage(),
            AddChanneld(keys=["image"]),  # Add channel dimension to the image
            Resized(keys=["image"], spatial_size=downsample_size, mode="trilinear"),  # Downsample image
            ScaleIntensityd(keys=['image']),  # Scale image intensities
            EnsureTyped(keys=["image", "classification_label"]),  # Ensure correct data types
        ]
    )

    return transforms

# -------------------------------------------------------
# Define CacheDataset and DataLoader for training and validation¶
# -------------------------------------------------------
def create_dataloaders(cfg,
                       data_dicts, 
                       transforms, 
                        ):
    # Get configuration parameters
    cache_rate_val = cfg['general'].get('cache_rate_val', 0.01)
    batch_size = cfg['general'].get('batch_size', 8)

    # Create datasets
    val_ds = CacheDataset(
        data=data_dicts,
        transform=transforms,
        cache_rate=cache_rate_val,
        num_workers=2
    )

    # Create validation DataLoader
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=4
    )

    return val_loader

# -------------------------------------------------------
# Create Model
# -------------------------------------------------------
def model_creater(cfg):

    model = ResNet(block=cfg['model'].get('block'),
                   layers=cfg['model'].get('layers'),
                   block_inplanes=cfg['model'].get('block_inplanes'),
                   spatial_dims=3,
                   n_input_channels=cfg['model'].get('in_channels'),
                   num_classes=2)
    

    print('creating the model ...')
    print(model)

    return model

# -------------------------------------------------------
# main function
# -------------------------------------------------------
def main(args):

    # -------------------------------------------------------
    # load config and create the root directory
    # -------------------------------------------------------
    config_path = args['config']
    with open(config_path, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    print('config: ', cfg)

    # -------------------------------------------------------
    # load dat dicts 
    # -------------------------------------------------------
    data_path_dir = args["data_dicts"]
        
    with open(data_path_dir, 'rb') as f:
        data_dicts_train, data_dicts_val, data_dicts_test = pickle.load(f)

    # with open(data_path_dir, 'rb') as f:
    #     data_dicts_test = pickle.load(f)

    data_dicts = data_dicts_test        
    print(data_dicts)
    print('selecting ', len(data_dicts), ' images as test dataset ...')

    # -------------------------------------------------------
    # define transforms 
    # -------------------------------------------------------

    transforms = train_val_transforms(cfg)   

    # -------------------------------------------------------
    # create dataset and dataloader
    # -------------------------------------------------------

    data_loader = create_dataloaders(cfg,
                                    data_dicts, 
                                    transforms)
    
    # -------------------------------------------------------
    # create model
    # -------------------------------------------------------

    model = model_creater(cfg)

    # -------------------------------------------------------
    # GPU selection
    # -------------------------------------------------------
    # os.environ["CUDA_VISIBLE_DEVICES"] = cfg['general'].get('GPU', '0')
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    device = torch.device("cuda:0")
    model.to(device)
    print('model loaded to the device')

    # -------------------------------------------------------
    # check if the model is trained ; just load the model
    # -------------------------------------------------------
    model_path = args['model_path']
    model.load_state_dict(torch.load(model_path))
    print('trained model loaded!')

    # -------------------------------------------------------
    # inference loop
    # -------------------------------------------------------
    val_step = 0
    all_val_preds = []
    all_val_labels = []

    model.eval()

    with torch.no_grad():
        for val_data in data_loader:
            val_step += 1

            # Move inputs and labels to the device
            val_inputs, val_labels = (
                val_data["image"].to(device),
                val_data["classification_label"].to(device),
            )

            # Get the image paths for this batch
            img_paths = val_data["img_path"]

            # Perform inference
            with torch.cuda.amp.autocast():
                val_outputs = model(val_inputs)

            # Convert logits to probabilities using softmax
            val_probs = torch.softmax(val_outputs, dim=1)  # Shape: (batch_size, 2)
            #val_preds = torch.argmax(val_probs, dim=1)  # Shape: (batch_size,)
            # Use the probability of the positive class (class 1)
            print(val_probs)
            val_preds = (val_probs[:, 1] > args["binarization_threshold"]).long()  # Shape: (batch_size,)


            # Move tensors to CPU for sklearn metrics
            val_preds_cpu = val_preds.cpu().numpy()
            val_labels_cpu = val_labels.cpu().numpy()

            # Iterate over each item in the batch
            for i in range(len(img_paths)):
                # Print the image path and corresponding label
                print(f"Image Path: {img_paths[i]}, Label: {val_labels_cpu[i]}, Prediction: {val_preds_cpu[i]}")

                all_val_preds.append(val_preds_cpu[i])
                all_val_labels.append(val_labels_cpu[i])

    print(f"gathered {len(all_val_preds)} predictions and {len(all_val_labels)} labels")
    # Calculate precision, recall, and F1 score for this item
    avg_prec = precision_score(all_val_labels, all_val_preds, zero_division=0)
    avg_sen = recall_score(all_val_labels, all_val_preds, zero_division=0)
    avg_f1 = f1_score(all_val_labels, all_val_preds, zero_division=0)

    print(
        f"\nAverage Recall: {avg_sen:.4f}",
        f"\nAverage Precision: {avg_prec:.4f}",
        f"\nAverage F1: {avg_f1:.4f}"
    )


if __name__ == '__main__':

    # -------------------------------------------------------
    # create parser
    # -------------------------------------------------------
    my_parser = argparse.ArgumentParser(description='Working directory')

    # Add the arguments
    my_parser.add_argument('-m','--model_path', help='path of trained model', required=True)
    my_parser.add_argument('-cfg','--config', help='path of config file', required=True)
    my_parser.add_argument('-d','--data_dicts', help='path of data_dicts', required=True)
    my_parser.add_argument('-b','--binarization_threshold', help='threshold to binarize the model; between 0-1', required=False, default=0.5, type=float)

    args = vars(my_parser.parse_args())

    main(args)

