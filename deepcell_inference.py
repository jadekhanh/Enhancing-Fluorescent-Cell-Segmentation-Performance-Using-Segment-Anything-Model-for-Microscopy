import argparse
import json
import os
import zarr 
import torch
import os
import zarr
from skimage.io import imsave
from glob import glob
from typing import List, Optional, Union, Tuple
from PIL import Image

import numpy as np
import pandas as pd

from segment_anything import SamPredictor
from tqdm import tqdm

from micro_sam.instance_segmentation import AutomaticMaskGenerator, load_instance_segmentation_with_decoder_from_checkpoint, default_grid_search_values_amg
from micro_sam import instance_segmentation, inference, evaluation
from micro_sam.evaluation.experiments import default_experiment_settings, full_experiment_settings
from micro_sam import util
from torch_em.data.datasets.dynamicnuclearnet import get_dynamicnuclearnet_loader

#
# Inference
#

def get_predictor(
    model_type: str,
    checkpoint_path: Union[str, os.PathLike],
    device: Optional[Union[str, torch.device]] = None,
) -> SamPredictor:
    """Load the SAM model (predictor) and instance segmentation decoder.

    This requires a checkpoint that contains the state for both predictor
    and decoder.

    Args:
        model_type: The type of the image encoder used in the SAM model.
        checkpoint_path: Path to the checkpoint from which to load the data.
        device: The device.

    Returns:
        The SAM predictor.
        The decoder for instance segmentation.
    """
    checkpoint_path = "/usr3/graduate/kontact/.cache/micro_sam/models/vit_h"
    model_type = "vit_h"
    device = torch.device("cuda")
    predictor = util.get_sam_model(
        model_type=model_type, checkpoint_path=checkpoint_path,
        device=device
    )
    return predictor


def save_image(image_array, image_path):
    image = Image.fromarray(image_array)
    image.save(image_path)


import os
import zarr

def _get_deepcell_paths(input_folder, split="test"):
    """
    Retrieve paths for raw and ground truth images from the DeepCell dataset.

    Args:
        input_folder (str): Path to the folder containing the DeepCell dataset.
        split (str, optional): Specifies the dataset split to use, either "val" or "test". Defaults to "test".

    Returns:
        list, list: Lists containing paths to raw images and ground truth images respectively.
    """
    assert split in ["val", "test"]

    gt_paths = []
    image_paths = []

    # Create the path to test and val folder
    output_folder = os.path.join("/projectnb/rfpm/SAM_kontact/fluorescent/datasets/DynamicNuclearNet", split)

    # Path to the dataset
    dataset_folder = os.path.join("/projectnb/rfpm/SAM_kontact/fluorescent/datasets/", split)
    
    # List all .zarr files in the dataset folder and sort them
    zarr_files = sorted([f for f in os.listdir(dataset_folder) if f.endswith(".zarr")])

    # Create ground truth and raw image folders: DynamicNuclearNet/test/gt and DynamicNuclearNet/test/image
    gt_folder = os.path.join(output_folder, "gt") # ground truth folder
    image_folder = os.path.join(output_folder, "image") # raw image folder

    # Create the output folders if they don't exist
    os.makedirs(gt_folder, exist_ok=True)
    os.makedirs(image_folder, exist_ok=True)

    # Iterate through each .zarr file in the dataset
    for i, zarr_file in enumerate(zarr_files):
        # Open the .zarr file
        with zarr.open(os.path.join(dataset_folder, zarr_file)) as f:
            
            # Extract raw and ground truth data from the .zarr file
            raw_data = f['raw'][:]
            label_data = f['labels'][:]
            
            # Save raw images inside "image" folder
            raw_image_path = os.path.join(image_folder, f"raw_image_{i}.png")
            # Assuming you have a function save_image defined elsewhere
            save_image(raw_data, raw_image_path)
            image_paths.append(raw_image_path)
        
            # Save label images inside "gt" folder
            label_image_path = os.path.join(gt_folder, f"label_image_{i}.png")
            # Assuming you have a function save_image defined elsewhere
            save_image(label_data, label_image_path)
            gt_paths.append(label_image_path)

    return image_paths, gt_paths



def deepcell_inference(
    checkpoint: Union[str, os.PathLike],
    input_folder: Union[str, os.PathLike],
    model_type: str,
    experiment_folder: Union[str, os.PathLike],
    use_points: bool,
    use_boxes: bool,
    n_positives: Optional[int] = None,
    n_negatives: Optional[int] = None,
    prompt_folder: Optional[Union[str, os.PathLike]] = None,
    predictor: Optional[SamPredictor] = None
) -> None:
    """Run inference for deepcell with a fixed prompt setting.

    Args:
        checkpoint: The segment anything model checkpoint.
        input_folder: The folder with the deepcell data.
        model_type: The type of the segment anything model.
        experiment_folder: The folder where to save all data associated with the experiment.
        use_points: Whether to use point prompts.
        use_boxes: Whether to use box prompts.
        n_positives: The number of positive point prompts.
        n_negatives: The number of negative point prompts.
        prompt_folder: The folder where the prompts should be saved.
        predictor: The segment anything predictor.
    """
    image_paths, gt_paths = _get_deepcell_paths(input_folder)
    if predictor is None:
        predictor = get_predictor(checkpoint, model_type)

    if use_boxes and use_points:
        assert (n_positives is not None) and (n_negatives is not None)
        setting_name = f"box/p{n_positives}-n{n_negatives}"
    elif use_boxes:
        setting_name = "box/p0-n0"
    elif use_points:
        assert (n_positives is not None) and (n_negatives is not None)
        setting_name = f"points/p{n_positives}-n{n_negatives}"
    else:
        raise ValueError("You need to use at least one of point or box prompts.")

    # we organize all folders with data from this experiment beneath 'experiment_folder'
    prediction_folder = os.path.join(experiment_folder, setting_name)  # where the predicted segmentations are saved
    os.makedirs(prediction_folder, exist_ok=True)
    embedding_folder = os.path.join(experiment_folder, "embeddings")  # where the precomputed embeddings are saved
    os.makedirs(embedding_folder, exist_ok=True)

    # NOTE: we can pass an external prompt folder, to make re-use prompts from another experiment
    # for reproducibility / fair comparison of results
    if prompt_folder is None:
        prompt_folder = os.path.join(experiment_folder, "prompts")
        os.makedirs(prompt_folder, exist_ok=True)

    inference.run_inference_with_prompts(
        predictor,
        image_paths,
        gt_paths,
        embedding_dir=embedding_folder,
        prediction_dir=prediction_folder,
        prompt_save_dir=prompt_folder,
        use_points=use_points,
        use_boxes=use_boxes,
        n_positives=n_positives,
        n_negatives=n_negatives,
    )


def run_deepcell_amg(
    checkpoint: Union[str, os.PathLike],
    input_folder: Union[str, os.PathLike],
    model_type: str,
    experiment_folder: Union[str, os.PathLike],
    iou_thresh_values: Optional[List[float]] = None,
    stability_score_values: Optional[List[float]] = None,
    verbose_gs: bool = False
) -> str:
    """Run automatic mask generation grid-search and inference for deepcell.

    Args:
        checkpoint: The segment anything model checkpoint.
        input_folder: The folder with the deepcell data.
        model_type: The type of the segmenta anything model.
        experiment_folder: The folder where to save all data associated with the experiment.
        iou_thresh_values: The values for `pred_iou_thresh` used in the gridsearch.
            By default values in the range from 0.6 to 0.9 with a stepsize of 0.025 will be used.
        stability_score_values: The values for `stability_score_thresh` used in the gridsearch.
            By default values in the range from 0.6 to 0.9 with a stepsize of 0.025 will be used.
        verbose_gs: Whether to run the gridsearch for individual images in a verbose mode.

    Returns:
        The path where the predicted images are stored.
    """
    embedding_folder = os.path.join(experiment_folder, "embeddings")  # where the precomputed embeddings are saved
    os.makedirs(embedding_folder, exist_ok=True)

    predictor = get_predictor(checkpoint, model_type)
    amg = AutomaticMaskGenerator(predictor)
    amg_prefix = "amg"

    # where the predictions are saved
    prediction_folder = os.path.join(experiment_folder, amg_prefix, "inference")
    os.makedirs(prediction_folder, exist_ok=True)

    # where the grid-search results are saved
    gs_result_folder = os.path.join(experiment_folder, amg_prefix, "grid_search")
    os.makedirs(gs_result_folder, exist_ok=True)

    val_image_paths, val_gt_paths = _get_deepcell_paths(input_folder, "val")
    test_image_paths, _ = _get_deepcell_paths(input_folder, "test")

    grid_search_values = instance_segmentation.default_grid_search_values_amg(
        iou_thresh_values=iou_thresh_values,
        stability_score_values=stability_score_values,
    )

    instance_segmentation.run_instance_segmentation_grid_search_and_inference(
        amg, grid_search_values,
        val_image_paths, val_gt_paths, test_image_paths,
        embedding_folder, prediction_folder, gs_result_folder,
    )
    return prediction_folder


def run_deepcell_instance_segmentation_with_decoder(
    checkpoint: Union[str, os.PathLike],
    input_folder: Union[str, os.PathLike],
    model_type: str,
    experiment_folder: Union[str, os.PathLike],
    verbose_gs: bool = False
) -> str:
    """Run automatic mask generation grid-search and inference for deepcell.

    Args:
        checkpoint: The segment anything model checkpoint.
        input_folder: The folder with the deepcell data.
        model_type: The type of the segmenta anything model.
        experiment_folder: The folder where to save all data associated with the experiment.
        verbose_gs: Whether to run the gridsearch for individual images in a verbose mode.

    Returns:
        The path where the predicted images are stored.
    """
    embedding_folder = os.path.join(experiment_folder, "embeddings")  # where the precomputed embeddings are saved
    os.makedirs(embedding_folder, exist_ok=True)

    segmenter = load_instance_segmentation_with_decoder_from_checkpoint(
        checkpoint, model_type,
    )
    seg_prefix = "instance_segmentation_with_decoder"

    # where the predictions are saved
    prediction_folder = os.path.join(experiment_folder, seg_prefix, "inference")
    os.makedirs(prediction_folder, exist_ok=True)

    # where the grid-search results are saved
    gs_result_folder = os.path.join(experiment_folder, seg_prefix, "grid_search")
    os.makedirs(gs_result_folder, exist_ok=True)

    val_image_paths, val_gt_paths = _get_deepcell_paths(input_folder, "val")
    test_image_paths, _ = _get_deepcell_paths(input_folder, "test")

    # get the default grid search values
    grid_search_values = instance_segmentation.default_grid_search_values_instance_segmentation_with_decoder()

    # generates predictions
    instance_segmentation.run_instance_segmentation_grid_search_and_inference(
        segmenter, grid_search_values,
        val_image_paths, val_gt_paths, test_image_paths,
        embedding_dir=embedding_folder, prediction_dir=prediction_folder,
        result_dir=gs_result_folder,
    )
    return prediction_folder


def _run_multiple_prompt_settings(checkpoint, input_folder, model_type, prompt_settings):
    predictor = get_predictor(checkpoint, model_type)
    for settings in prompt_settings:
        deepcell_inference(
            checkpoint,
            input_folder,
            model_type,
            "/projectnb/rfpm/SAM_kontact/fluorescent/experiment", # experiment folder
            use_points=settings["use_points"],
            use_boxes=settings["use_boxes"],
            n_positives=settings["n_positives"],
            n_negatives=settings["n_negatives"],
            prompt_folder=None,
            predictor=predictor
        )


def run_deepcell_inference() -> None:
    """Run deepcell inference."""

    # using fine-tuned full model vit_h model
    checkpoint = "/projectnb/rfpm/SAM_kontact/fluorescent/checkpoints/deepcell_sam_full_model_vit_h/best.pt"
    input_folder = "/projectnb/rfpm/SAM_kontact/fluorescent/datasets"
    model_type = "vit_h"
    experiment_folder = "/projectnb/rfpm/SAM_kontact/fluorescent/experiment"

    # Uncomment this code block to perform interactive segmentation
    prompt_settings = default_experiment_settings()
    _run_multiple_prompt_settings(checkpoint, input_folder, model_type, prompt_settings)

    # Uncomment this to perform automatic segmentation
    # run_deepcell_amg(checkpoint, input_folder, model_type, experiment_folder)


if __name__ == "__main__":

    run_deepcell_inference()
