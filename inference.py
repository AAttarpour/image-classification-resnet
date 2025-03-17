"""
This code is written by Ahmadreza Attarpour (a.attarpour@mail.utoronto.ca)
This is the inference code for the patch classification task.
The code loads the trained model and performs inference on the test dataset.

input:
    - model_path: path of the trained model
    - config: path of the config file
    - binarization_threshold: threshold to binarize the model ; between 0-1
    - patch_path: path to the patches

output:
    - metadata file with the predicted labels and the corresponding image paths
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
from tqdm import tqdm
import json
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
                image_dict['img_file_name'] = os.path.basename(image_dict['image'])
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
            EnsureTyped(keys=["image"]),  # Ensure correct data types
        ]
    )

    return transforms

# -------------------------------------------------------
# Define CacheDataset and DataLoader for training and validation¶
# -------------------------------------------------------
def create_dataloaders(cfg,
                       data_dicts, 
                       transforms, 
                       batch_size = 10
                        ):
    # Get configuration parameters
    cache_rate_val = cfg['general'].get('cache_rate_val', 0.01)
    batch_size = batch_size

    # Create datasets
    val_ds = CacheDataset(
        data=data_dicts,
        transform=transforms,
        cache_rate=cache_rate_val,
        num_workers=4
    )

    # Create validation DataLoader
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=10
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
    

    print('model created!')

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
    input_path = args['input_path']
    images_val = []
    with os.scandir(input_path) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.lower().endswith((".tif", ".tiff")):
                images_val.append(os.path.join(input_path, entry.name))

    data_dicts = [{"image": image_name} for image_name in images_val]
    print("Data dicts are ready!")
    print(f"Some samples from data dicts: {data_dicts[:3]}")
    # -------------------------------------------------------
    # define transforms 
    # -------------------------------------------------------

    transforms = train_val_transforms(cfg)   

    # -------------------------------------------------------
    # create dataset and dataloader
    # -------------------------------------------------------

    batch_size = args['batch_size']
    data_loader = create_dataloaders(cfg,
                                    data_dicts, 
                                    transforms,
                                    batch_size = batch_size
                                    )
    
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
    # Initialize metadata dictionary
    metadata = {
        'patches': []
    }
    val_step = 0

    model.eval()

    with torch.no_grad():
        for val_data in tqdm(data_loader):
            val_step += 1
            
            # Move inputs and labels to the device
            val_inputs = val_data["image"].to(device)


            # Get the image paths for this batch
            img_file_names = val_data["img_file_name"]

            # Perform inference
            with torch.cuda.amp.autocast():
                val_outputs = model(val_inputs)

            # Convert logits to probabilities using softmax
            val_probs = torch.softmax(val_outputs, dim=1)  # Shape: (batch_size, 2)
            #val_preds = torch.argmax(val_probs, dim=1)  # Shape: (batch_size,)
            # Use the probability of the positive class (class 1)
            val_preds = (val_probs[:, 1] > args["binarization_threshold"]).long()  # Shape: (batch_size,)

            # Move tensors to CPU for sklearn metrics
            val_preds_cpu = val_preds.cpu().numpy()

            # Iterate over each item in the batch
            for i in range(len(img_file_names)):

                # Append the image path, and prediction to the metadata dictionary
                results = {
                    'filename': img_file_names[i],
                    'classification_label': int(val_preds_cpu[i])
                }
                metadata['patches'].append(results)

            # tqdm.write(
            #     f"{val_step}/{len(data_loader.dataset) // batch_size}"
            #     )

    # Save the metadata dictionary to a JSON file
    with open(os.path.join(input_path, 'metadata_patch_classification.json'), 'w') as f:
        json.dump(metadata, f, indent=4)


if __name__ == '__main__':

    # -------------------------------------------------------
    # create parser
    # -------------------------------------------------------
    my_parser = argparse.ArgumentParser(description='Working directory')

    # Add the arguments
    my_parser.add_argument('-m','--model_path', help='path of trained model', required=True)
    my_parser.add_argument('-c','--config', help='path of config file', required=True)
    my_parser.add_argument('-i','--input_path', help='path to patches', required=True)
    my_parser.add_argument('-t','--binarization_threshold', help='threshold to binarize the model ; between 0-1', required=False, default=0.5, type=float)
    my_parser.add_argument('-b','--batch_size', help='batch size', required=False, default=10, type=int)

    

    args = vars(my_parser.parse_args())

    main(args)

