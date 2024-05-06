import os
import numpy as np
from math import ceil, floor

import torch
import torch_em
import imageio
import cv2
import torch_em.data.datasets as datasets
from torch_em.transform.label import label_consecutive
from torch_em.data import MinInstanceSampler, ConcatDataset
from torch_em.transform.raw import standardize, normalize_percentile
from torch.utils.data import Dataset, DataLoader
from torch_em.transform.raw import normalize_percentile, normalize



def raw_transform_rgb(raw):
    # Check if the input data has three channels
    if raw.shape[-1] == 3:
        # If it has three channels, convert it to grayscale
        raw = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)
    # # Ensure the image shape is (512, 512)
    # raw = cv2.resize(raw, (512, 512))
    # Apply other transformations
    raw = normalize_percentile(raw)
    raw = normalize(raw)
    raw = raw * 255
    return raw


def raw_trasform_grayscale(raw):
    raw = normalize(raw)
    raw = raw * 225
    return raw

def convert_images_to_2d(directory):
    """
    If all images are 3D, squeeze them into 2D images
    """
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        print("file path:", filepath)
        if os.path.isdir(os.path.join(directory, filename)) or filename.startswith('.'):
            continue
        image = cv2.imread(filepath)
        image_2d = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        cv2.imwrite(filepath, image_2d)

def get_dataset(input_path, patch_shape, split_choice, raw_key, label_key):
    """Return train or val data loader from .jpg, .png, and .tif datasets for finetuning SAM.

    The data loader must be a torch data loader that retuns `x, y` tensors,
    where `x` is the image data and `y` are the labels.
    The labels have to be in a label mask instance segmentation format.
    I.e. a tensor of the same spatial shape as `x`, with each object mask having its own ID.
    Important: the ID 0 is reseved for background, and the IDs must be consecutive

    Here, we use `torch_em.default_segmentation_loader` for creating a suitable data loader from
    the data.
    """
    assert split_choice in ("train", "val")
    batch_size = 1

    # Define the directories containing images and masks for training and validation
    train_image_dir = os.path.join(input_path, "train", "images")
    train_mask_dir = os.path.join(input_path, "train", "masks")
    val_image_dir = os.path.join(input_path, "val", "images")
    val_mask_dir = os.path.join(input_path, "val", "masks")

    # Load images and masks from the specified directories
    if split_choice == "train":
        image_dir = train_image_dir
        segmentation_dir = train_mask_dir
    else:
        image_dir = val_image_dir
        segmentation_dir = val_mask_dir

    # Check if the provided paths are correct or not
    print(len(os.path.join(image_dir, raw_key)))
    print(len(os.path.join(segmentation_dir, label_key)))
    
    # The 'roi' argument can be used to subselect parts of the data.
    # Here, we use it to select the first 70 frames fro the test split and the other frames for the val split.
    # If you don't need ROI selection, you can remove this part.
    roi = None

    # Initialize a MinInstanceSampler for sampling instances from the dataset
    sampler = MinInstanceSampler(min_num_instances=3)

    if train_instance_segmentation:
        # Computes the distance transform for objects to perform end-to-end automatic instance segmentation.
        label_transform = PerObjectDistanceTransform(
            distances=True, boundary_distances=True, directed_distances=False,
            foreground=True, instances=True, min_size=25
        )
    else:
        label_transform = torch_em.transform.label.connected_components
        
    loader = torch_em.default_segmentation_loader(
        raw_paths=image_dir, raw_key=raw_key,
        label_paths=segmentation_dir, label_key=label_key,
        patch_shape=patch_shape, batch_size=batch_size,
        ndim=2, is_seg_dataset=True, rois=roi,
        raw_transform=raw_transform_rgb,
        label_transform=label_transform,
        num_workers=8, shuffle=True, sampler=sampler
    )

    return loader

    
def get_concat_fluorescent_datasets(input_path, patch_shape, split_choice):
    """
    Returns a concatenated dataset containing data from multiple electron microscopy datasets.
    
    Args:
        input_path (str): Path to the input datasets.
        patch_shape (tuple): Patch shape for the datasets.
        
    Returns:
        tuple: Tuple containing train and validation datasets.
    """
    # Assert that the split choice is either "train" or "val"
    assert split_choice in ["train", "val"]

    # Get DeepCELL dataset
    deepcell = get_dataset(os.path.join(input_path, "deepcell"), patch_shape, split_choice, raw_key = "*.png", label_key = "*.png")

    # # Get Fluorescent Neuronal Cells v2 (FNC v2) dataset
    # fncv2 = get_dataset(os.path.join(input_path, "fncv2"), patch_shape, split_choice, raw_key = "*.png", label_key = "*.png")

    # # Get Automatic dataset
    # automatic = get_dataset(os.path.join(input_path, "automatic"), patch_shape, split_choice, raw_key = "*.png", label_key = "*.png")

    # # Get BitDepth dataset
    # bitdepth = get_dataset(os.path.join(input_path, "bitdepth"), patch_shape, split_choice, raw_key = "*.tif", label_key = "*.png")

    # # Get Cellpose dataset
    # cellpose = get_dataset(os.path.join(input_path, "cellpose"), patch_shape, split_choice, raw_key = "*.png", label_key = "*.png")

    # # Get Fluorescence Microscopy Denoising (FMD) dataset
    # fmd = get_dataset(os.path.join(input_path, "fmd"), patch_shape, split_choice, raw_key = "*.png", label_key = "*.png")

    # # Get Fluorescent Neuronal Cells (FNC) dataset
    # fnc = get_dataset(os.path.join(input_path, "fnc"), patch_shape, split_choice, raw_key = "*.png", label_key = "*.png")

    # # Get HL60 dataset
    # hl60 = get_dataset(os.path.join(input_path, "hl60"), patch_shape, split_choice, raw_key = "*.tif", label_key = "*.tif")

    # # Get Hoechst dataset
    # hoechst = get_dataset(os.path.join(input_path, "hoechst"), patch_shape, split_choice, raw_key = "*.png", label_key = "*.png")

    # # Get Kaggle 2018 dataset
    # kaggle2018 = get_dataset(os.path.join(input_path, "kaggle2018"), patch_shape, split_choice, raw_key = "*.tif", label_key = "*.tif")

    # # Get Practical dataset
    # practical = get_dataset(os.path.join(input_path, "practical"), patch_shape, split_choice, raw_key = "*.tif", label_key = "*.tif")

    # # Get StarDist dataset
    # stardist = get_dataset(os.path.join(input_path, "stardist"), patch_shape, split_choice, raw_key = "*.tif", label_key = "*.tif")

    # # Get Synthetic dataset
    # synthetic = get_dataset(os.path.join(input_path, "synthetic"), patch_shape, split_choice, raw_key = "*.tif", label_key = "*.tif")

    # # Get U2OS dataset
    # u2os = get_dataset(os.path.join(input_path, "u2os"), patch_shape, split_choice, raw_key = "*.tif", label_key = "*.png")
    
    # Extract datasets from DataLoader objects and concat them
    # fluorescent_dataset = ConcatDataset(deepcell.dataset, fncv2.dataset, automatic.dataset, bitdepth.dataset, cellpose.dataset, fmd.dataset, fnc.dataset, hl60.dataset, hoechst.dataset, kaggle2018.dataset, practical.dataset, stardist.dataset, synthetic.dataset, u2os.dataset)
    fluorescent_dataset = ConcatDataset(deepcell.dataset)

    # Return the concatenated dataset
    return fluorescent_dataset


def get_fluorescent_dataloaders(input_path, patch_shape):
    """
    Returns data loaders for the concatenated fluorescent microscopy datasets for finetuning.
    
    Args:
        input_path (str): Path to the input datasets.
        patch_shape (tuple): Patch shape for the datasets.
        
    Returns:
        tuple: Tuple containing train and validation data loaders.
    """
    fluorescent_train_dataset = get_concat_fluorescent_datasets(input_path, patch_shape, "train")
    fluorescent_val_dataset = get_concat_fluorescent_datasets(input_path, patch_shape, "val")
    train_loader = torch_em.get_data_loader(fluorescent_train_dataset, batch_size=2, shuffle=True, num_workers=16)
    val_loader = torch_em.get_data_loader(fluorescent_val_dataset, batch_size=1, shuffle=True, num_workers=16)
    return train_loader, val_loader
