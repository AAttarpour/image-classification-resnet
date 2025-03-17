"""
This code is written by Ahmadreza Attarpour (a.attarpour@mail.utoronto.ca)
It trains a classification model using the provided data and config file

The code is written in Pytorch and MONAI library is used for data loading and augmentation

main inputs:
    -exp: experiments directory
    -cfg: path of config file

output:
    - trained model
    - training and validation curves
    - best model based on F1 score
    - best model based on weighted F1 score

"""

# -------------------------------------------------------
# load libraries
# -------------------------------------------------------
import os
import pickle
from sched import scheduler
from monai.utils import set_determinism
import numpy as np
from monai.data import CacheDataset, DataLoader
from monai.losses import FocalLoss
from monai.transforms import (
    Compose,
    ScaleIntensityd,
    EnsureTyped,
    AddChanneld,
    RandAdjustContrastd,
    OneOf,
    RandGaussianSharpend,
    RandGaussianSmoothd,
    RandGaussianNoised,
    ToTensord,
    RandShiftIntensityd,
    RandHistogramShiftd,
    RandAxisFlipd,
    Transform,
    Resized,
    RandZoomd,
    RandRotated
)
import torch
import torch.nn as nn
import argparse
from datetime import datetime
from torchio.transforms import RandomAffine
from skimage.util import random_noise
import shutil
import yaml
from utils import WarmupCosineScheduler
import tifffile
from monai.networks.nets import ResNet
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
set_determinism(seed=0)


# -------------------------------------------------------
# Define transforms for image and segmentation
# ------------------------------------------------------- 

def train_val_transforms(cfg):
    # Get configuration parameters
    temp_value= cfg['general'].get('main_input_img_size', 256)
    main_input_img_size = (temp_value, temp_value, temp_value) # the main input to the model

    temp_value= cfg['general'].get('downsample_size', 128)
    downsample_size = (temp_value, temp_value, temp_value) # the main input to the model
    
    data_aug_prob = cfg['general'].get('data_aug_prob', 0.1)  # Probability of applying augmentation

    # image loader
    class MyLoadImage(Transform):
        def __call__(self, image_dict):
            # load the .tiff files they are HxWxD
            if isinstance(image_dict['image'], str):
                image_dict['image'] = tifffile.imread(image_dict['image'])

                # change the order axis of the image from DHW to HWD
                image_dict['image'] = np.moveaxis(image_dict['image'], 0, 2)
                image_dict['image'] = np.moveaxis(image_dict['image'], 0, 1)

                if "label" in image_dict:
                    del image_dict["label"]
                # add a label key to the dictionary required for randomaffine
                # image_dict['label'] = np.zeros(image_dict['image'].shape, dtype=np.uint8)

            return image_dict

    # class RemoveKey(Transform):
    #     def __call__(self, image_dict):

    #         del image_dict['label']            

    #         return image_dict
        
    # # Custom transform to convert segmentation GT to binary classification label
    # class SegmentationToClassification(Transform):
    #     def __init__(self, threshold=0.1):
    #         self.threshold = threshold

    #     def __call__(self, image_dict):
    #         # Check if the segmentation mask contains any object
    #         if np.sum(image_dict['label'] > self.threshold * image_dict['label'].size):
    #             image_dict['classification_label'] = 1  # Object present
    #         else:
    #             image_dict['classification_label'] = 0  # No object

    #         # Remove the segmentation label since it's no longer needed
    #         del image_dict['label']
    #         return image_dict

    # Custom transform to add salt-and-pepper noise
    class RandomSaltPepper(Transform):
        def __call__(self, image_dict):
            image_dict["image"] = torch.tensor(random_noise(image_dict["image"], mode='s&p', amount=0.001, clip=True))
            return image_dict

    # # Training transforms
    train_transforms = Compose(
        [
            MyLoadImage(),
            AddChanneld(keys=["image"]),  # Add channel dimension to the image
            Resized(keys=["image"], spatial_size=downsample_size, mode=("trilinear")),  # Downsample image
            ScaleIntensityd(keys=['image']),  # Scale image intensities and move the labels between 0 and 1
            OneOf(transforms=[
                # RandomAffine(include=["image", "label"], p=data_aug_prob, degrees=(30,30,30),
                #             scales=(0.5, 2), translation=(0.1,0.1,0.1),
                #             default_pad_value='mean', label_keys='label'),
                RandRotated(
                        keys=["image"],
                        range_x=30 * np.pi / 180,  # Rotate in the XY plane (30 degrees in radians)
                        range_y=30 * np.pi / 180,  # Rotate in the XZ plane (30 degrees in radians)
                        range_z=30 * np.pi / 180,  # Rotate in the YZ plane (30 degrees in radians)
                        prob=data_aug_prob,  # Probability of applying rotation
                        keep_size=True,  # Keep the output size the same as the input
                        mode="bilinear",  # Interpolation mode
                        padding_mode="border",  # Padding mode for outside grid values
                    ),
                RandZoomd(keys=["image"], prob=data_aug_prob, min_zoom=0.8, max_zoom=1.2, mode="trilinear"),
                RandomSaltPepper(),
                RandShiftIntensityd(keys=['image'], offsets=0.1, prob=data_aug_prob),
                RandAdjustContrastd(keys=["image"], prob=data_aug_prob, gamma=(0.5, 4)),
                RandGaussianSharpend(keys=["image"], prob=data_aug_prob),
                RandGaussianSmoothd(keys=["image"], prob=data_aug_prob),
                RandGaussianNoised(keys=["image"], prob=data_aug_prob, std=0.02),
                RandHistogramShiftd(keys=["image"], num_control_points=10, prob=data_aug_prob),
            ]),
            ToTensord(keys=["image"]),  # Convert image to tensor
            RandAxisFlipd(keys=["image"], prob=data_aug_prob),  # Randomly flip the image 
            # SegmentationToClassification(threshold=0.01),  # Convert segmentation GT to classification label # threshold=1 percent of the patch
            # RemoveKey(),
            EnsureTyped(keys=["image", "classification_label"]),  # Ensure correct data types
        ]
    )

    # Validation transforms
    val_transforms = Compose(
        [
            MyLoadImage(),
            # RemoveKey(),
            AddChanneld(keys=["image"]),  # Add channel dimension to the image
            Resized(keys=["image"], spatial_size=downsample_size, mode="trilinear"),  # Downsample image
            ScaleIntensityd(keys=['image']),  # Scale image intensities
            # SegmentationToClassification(threshold=0.01),  # Convert segmentation GT to classification label
            EnsureTyped(keys=["image", "classification_label"]),  # Ensure correct data types
        ]
    )

    return train_transforms, val_transforms

# -------------------------------------------------------
# Define CacheDataset and DataLoader for training and validation¶
# -------------------------------------------------------

def create_dataloaders(cfg,
                       data_dicts_train, 
                       train_transforms,
                       data_dicts_val,
                       val_transforms):
    # Get configuration parameters
    sampler_type = cfg['general'].get('dataloader_sampler', "default")
    cache_rate_train = cfg['general'].get('cache_rate_train', 0.0)
    cache_rate_val = cfg['general'].get('cache_rate_val', 0.01)
    batch_size = cfg['general'].get('batch_size', 8)

    # Create datasets
    train_ds = CacheDataset(
        data=data_dicts_train,
        transform=train_transforms,
        cache_rate=cache_rate_train,
        num_workers=2
    )
    val_ds = CacheDataset(
        data=data_dicts_val,
        transform=val_transforms,
        cache_rate=cache_rate_val,
        num_workers=2
    )

    # Create validation DataLoader
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=4
    )

    # Create training DataLoader
    if sampler_type == "default":
        print("Default DataLoader will be used ...")
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=10
        )
    else:
        print("Weighted Random Sampler will be used ...")

        # Calculate sample weights based on binary classification labels
        # Assuming the classification label is stored in the dictionary as 'classification_label'
        classification_labels = [img["classification_label"] for img in data_dicts_train]
        class_counts = np.bincount(classification_labels)
        class_weights = 1.0 / class_counts  # Inverse frequency weighting
        sample_weights = torch.tensor([class_weights[label] for label in classification_labels])

        print(f"Class counts: {class_counts}")
        print(f"Class weights: {class_weights}")
        print(f"Sample weights range: {sample_weights.min()} to {sample_weights.max()}")

        # Create WeightedRandomSampler
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_ds),
            replacement=True
        )

        # Create training DataLoader with WeightedRandomSampler
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=10
        )

    return train_loader, val_loader

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
# loss function
# -------------------------------------------------------
def loss_creater(cfg, device):

    loss_name = cfg['loss'].get('name')
    if loss_name == 'ce': 
        weight = torch.tensor([1, (cfg['loss'].get('ce_weights', 1.0))], dtype=torch.float, device=device)
        loss_function = torch.nn.CrossEntropyLoss(weight=weight, reduction="mean")

    elif loss_name == 'focal':
        loss_function = FocalLoss(to_onehot_y=True,  weight=weight, reduction="mean")

    return loss_function

# -------------------------------------------------------
# optimizer
# -------------------------------------------------------
def optimizer_creater(cfg, model):

    max_epochs = cfg['general'].get('max_epochs', 200)
    warmup_epochs = cfg['general'].get('warmup_epochs', 'default')
    if warmup_epochs == 'default':
        warmup_epochs = int(0.1 * max_epochs)
    learning_rate = cfg['optimizer'].get('learning_rate', 1e-4)
    weight_decay = cfg['optimizer'].get('weight_decay', 0.0)
    scheduler_last_epoch = cfg['optimizer'].get('scheduler_last_epoch', -1)
    optimizer_name = cfg['optimizer'].get('name', "Adam")
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': learning_rate}], weight_decay=weight_decay)
    elif optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW([{'params': model.parameters(), 'lr': learning_rate}], weight_decay=weight_decay)

    # if continuing the training, initialize the learining rate
    if scheduler_last_epoch != -1:
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
    # applying cosine algorithm (increase and then decrease the lr)
    scheduler = WarmupCosineScheduler(optimizer= optimizer, warmup_epochs=warmup_epochs, max_lr=learning_rate, max_epochs=max_epochs, last_epoch=scheduler_last_epoch) 


    return optimizer, scheduler

# -------------------------------------------------------
# plot training and metric curves
# -------------------------------------------------------
def plot_and_save_curves(epoch_loss_values, prec_list, sen_list, f1_list, root_dir):
    """
    Plots and saves the training loss, precision, recall, and F1 score curves.
    """
    # Set Seaborn style for better aesthetics
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))

    # Plot training loss
    plt.subplot(2, 2, 1)
    plt.plot(epoch_loss_values, label="Training Loss", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()

    # Plot precision
    plt.subplot(2, 2, 2)
    plt.plot(prec_list, label="Precision", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.title("Precision Curve")
    plt.legend()

    # Plot recall
    plt.subplot(2, 2, 3)
    plt.plot(sen_list, label="Recall", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.title("Recall Curve")
    plt.legend()

    # Plot F1 score
    plt.subplot(2, 2, 4)
    plt.plot(f1_list, label="F1 Score", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("F1 Score Curve")
    plt.legend()

    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(root_dir, "training_metrics_curves.png"))
    plt.close()

# -------------------------------------------------------
# weighted F1
# -------------------------------------------------------
def compute_metric(avg_sen, avg_prec, beta=2, epsilon=1e-10):
    '''
    compute weighted F1 score
    A β value greater than 1 favors the recall metric, 
    while values lower than 1 favor the precision metric.
    '''
    numerator = (1 + beta**2) * avg_sen * avg_prec
    denominator = ((beta**2) * avg_prec) + avg_sen + epsilon  # Adding epsilon to avoid division by zero
    return numerator / denominator


# -------------------------------------------------------
# main function
# -------------------------------------------------------
def main(args):

    # -------------------------------------------------------
    # load config and create the root directory
    # -------------------------------------------------------

    # load config file
    config_path = args['config']
    with open(config_path, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    print('config: ', cfg)

    # root_dir is where results will be stored
    root_dir = os.path.join(args['experiment'], cfg['general'].get('experiment_name') + '-' + datetime.today().strftime('%Y-%m-%d'))
    isExist = os.path.exists(root_dir)
    if not isExist: os.mkdir(root_dir)

    # copy the config file to the result directory for further references
    orig = os.path.basename(config_path)
    tar = os.path.join(root_dir, orig)
    shutil.copy2(config_path, tar)


    # -------------------------------------------------------
    # load dat dicts 
    # -------------------------------------------------------
    data_path_dir = cfg['general'].get('data_dict_file', True)
        
    with open(data_path_dir, 'rb') as f:
        data_dicts_train, data_dicts_val, data_dicts_test = pickle.load(f)

    # -------------------------------------------------------
    # temp code add new data_dicts
    # with open("/data3/projects/ahmadreza/DeepTrace/patch_classification/newly_patch_classification_data_dicts.pickle", 'rb') as f:
    #     data_dicts = pickle.load(f)

    # # shuffle the data_dicts
    # np.random.shuffle(data_dicts)
    # # add it to the data_dicts_train and data_dicts_val and data_dicts_test
    # data_dicts_train = data_dicts[:int(0.8*len(data_dicts))] + data_dicts_train
    # data_dicts_val = data_dicts[int(0.8*len(data_dicts)):int(0.9*len(data_dicts))] + data_dicts_val
    # data_dicts_test = data_dicts[int(0.9*len(data_dicts)):] + data_dicts_test
    # -------------------------------------------------------

    # saving data_dicts for inference code
    with open(os.path.join(root_dir, "data_dicts.pickle"), 'wb') as f:
        pickle.dump([data_dicts_train, data_dicts_val, data_dicts_test], f)  
        
    print('selecting ', len(data_dicts_train), ' images as train dataset ...')
    print('selecting ', len(data_dicts_val), ' images as validation dataset ...')
    print('selecting ', len(data_dicts_test), ' images as test dataset ...')

    # -------------------------------------------------------
    # define train and val transforms 
    # -------------------------------------------------------
    train_transforms, val_transforms = train_val_transforms(cfg)


    # -------------------------------------------------------
    # create dataset and dataloader
    # -------------------------------------------------------

    train_loader, val_loader = create_dataloaders(cfg,
                                                data_dicts_train, 
                                                train_transforms,
                                                data_dicts_val,
                                                val_transforms)
    
    # -------------------------------------------------------
    # create model
    # -------------------------------------------------------

    model = model_creater(cfg)

    # -------------------------------------------------------
    # GPU selection
    # -------------------------------------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg['general'].get('GPU', '0')
    device = torch.device("cuda:0")
    model.to(device)
    print('model loaded to the device')

    # -------------------------------------------------------
    # loss function 
    # -------------------------------------------------------

    loss_function = loss_creater(cfg, device)

    # -------------------------------------------------------
    # optimizer and scheduler
    # -------------------------------------------------------

    optimizer, scheduler = optimizer_creater(cfg, model)

    # -------------------------------------------------------
    # check if the model is trained ; just load the model
    # -------------------------------------------------------
    model_state = cfg['general'].get('model_state', 'untrained')
    model_path = cfg['general'].get('model_trained_path')
    if model_state == 'trained':
        model.load_state_dict(torch.load(model_path))
        print('trained model loaded!')

    # -------------------------------------------------------
    # training loop
    # -------------------------------------------------------

    batch_size = cfg['general'].get('batch_size', 8)
    max_epochs = cfg['general'].get('max_epochs', 200)
    val_interval = cfg['general'].get('validation_interval', 2)
    save_result_interval = cfg['general'].get('checkpoint_period', 10)
    gradient_clipping = cfg['optimizer'].get('grad_clip', 1)

    early_stop_patience_cnt = 0
    best_metric = -1
    best_metric_epoch = -1
    best_f1 = -1
    best_f1_epoch = -1
    epoch_loss_values = []
    epoch_val_f1_values = []
    epoch_val_prec_values = []
    epoch_val_recall_values = []

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(max_epochs):

        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        print(f'learning rate to {scheduler.get_lr()}')

        model.train()

        epoch_loss = 0
        step = 0
        iters = len(train_loader)

        for batch_data in train_loader:

            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["classification_label"].to(device),
            ) 

            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(inputs)

                loss = loss_function(outputs.float(), labels)

            scaler.scale(loss).backward()
    
            # gradient clipping
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step(epoch + step / iters)
            epoch_loss += loss.item()

            step += 1
            print(f"{step}/{len(train_loader)}, train_loss: {loss.item():.4f}")

            
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        
         # Validation step
        if (epoch + 1) % val_interval == 0:
            val_step = 0
            all_val_preds = []
            all_val_labels = []

            model.eval()

            with torch.no_grad():
                for val_data in val_loader:
                    val_step += 1

                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["classification_label"].to(device),
                    )

                    with torch.cuda.amp.autocast():
                        val_outputs = model(val_inputs)

                    # Convert logits to probabilities using softmax
                    val_probs = torch.softmax(val_outputs, dim=1)  # Shape: (batch_size, 2)
                    #val_preds = torch.argmax(val_probs, dim=1)  # Shape: (batch_size,)
                    # Use the probability of the positive class (class 1)
                    val_preds = (val_probs[:, 1] > 0.5).long()  # Shape: (batch_size,)

                    # Move tensors to CPU for sklearn metrics
                    val_preds_cpu = val_preds.cpu().numpy()
                    val_labels_cpu = val_labels.cpu().numpy()

                    # Iterate over each item in the batch
                    for i in range(len(val_preds)):
                        # Print the image path and corresponding label
                        all_val_preds.append(val_preds_cpu[i])
                        all_val_labels.append(val_labels_cpu[i])

            # Calculate precision, recall, and F1 score for this item
            avg_prec = precision_score(all_val_labels, all_val_preds, zero_division=0)
            avg_sen = recall_score(all_val_labels, all_val_preds, zero_division=0)
            avg_f1 = f1_score(all_val_labels, all_val_preds, zero_division=0)

            epoch_val_f1_values.append(avg_f1)
            epoch_val_prec_values.append(avg_prec)
            epoch_val_recall_values.append(avg_sen)

            print(
                f"\nAverage Recall: {avg_sen:.4f}",
                f"\nAverage Precision: {avg_prec:.4f}",
                f"\nAverage F1: {avg_f1:.4f}"
            )

            # Calculate the weighted F1 score
            metric = compute_metric(avg_sen, avg_prec)
            # Save the best model based on F1 score
            if (metric >= best_metric):
                best_metric = metric
                best_metric_epoch = epoch + 1
                early_stop_patience_cnt = 0
                torch.save(model.state_dict(), os.path.join(
                    root_dir, f"best_metric_model_e{epoch + 1}.pth"))
                print("saved new best metric model")
            else:
                early_stop_patience_cnt += 1
            
            if (avg_f1 > best_f1):
                best_f1 = avg_f1
                best_f1_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(
                    root_dir, f"best_f1_model_e{epoch + 1}.pth"))
                print("saved new best f1 model")

            # Early stopping
            if early_stop_patience_cnt >= cfg['general'].get('early_stop_patience', 10):
                break

            print(
                f"current epoch: {epoch + 1} current f1: {avg_f1:.4f}"
                f"\nbest metric: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}"
                f"\nbest f1: {best_f1:.4f}"
                f" at epoch: {best_f1_epoch}"
            )


        # checkpoint period   
        if (epoch + 1) % save_result_interval == 0:

            plot_and_save_curves(epoch_loss_values, epoch_val_prec_values, epoch_val_recall_values, epoch_val_f1_values, root_dir)
            
            with open(os.path.join(root_dir, "results.pickle"), 'wb') as f:
                pickle.dump([epoch_loss_values, epoch_val_prec_values, epoch_val_recall_values, epoch_val_f1_values], f)
            
            torch.save(model.state_dict(), os.path.join(root_dir, "last_trained_model.pth"))
            
    print(
        f"train completed, best_metric: {best_metric:.4f} "
        f"at epoch: {best_metric_epoch}")

    # -------------------------------------------------------
    # saving epoch_loss_values and metric_values and last trained model
    # -------------------------------------------------------
    with open(os.path.join(root_dir, "results.pickle"), 'wb') as f:
        pickle.dump([epoch_loss_values, epoch_val_prec_values, epoch_val_recall_values, epoch_val_f1_values], f)

    plot_and_save_curves(epoch_loss_values, epoch_val_prec_values, epoch_val_recall_values, epoch_val_f1_values, root_dir)

    torch.save(model.state_dict(), os.path.join(root_dir, "last_trained_model.pth"))
    print("saved last trained model")

if __name__ == '__main__':

    # -------------------------------------------------------
    # create parser
    # -------------------------------------------------------
    my_parser = argparse.ArgumentParser(description='Working directory')

    # Add the arguments
    my_parser.add_argument('-exp','--experiment', help='experiments directory', required=True)
    my_parser.add_argument('-cfg','--config', help='path of config file', required=True)
    args = vars(my_parser.parse_args())

    main(args)