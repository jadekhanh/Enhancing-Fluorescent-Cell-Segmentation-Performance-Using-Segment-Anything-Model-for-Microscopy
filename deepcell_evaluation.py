"""Evaluation functionality for segmentation predictions from `micro_sam.evaluation.automatic_mask_generation`
and `micro_sam.evaluation.inference`.
"""

import os
from pathlib import Path
from typing import List, Optional, Union

import imageio.v3 as imageio
import numpy as np
import pandas as pd
import zarr

from elf.evaluation import mean_segmentation_accuracy
from skimage.measure import label
from tqdm import tqdm


def _run_evaluation(gt_paths, prediction_paths, verbose=True):
    assert len(gt_paths) == len(prediction_paths)
    msas, sa50s, sa75s = [], [], []

    for gt_path, pred_path in tqdm(
        zip(gt_paths, prediction_paths), desc="Evaluate predictions", total=len(gt_paths), disable=not verbose
    ):
        assert os.path.exists(gt_path), gt_path
        assert os.path.exists(pred_path), pred_path

        gt = imageio.imread(gt_path)
        gt = label(gt)
        pred = imageio.imread(pred_path)

        msa, scores = mean_segmentation_accuracy(pred, gt, return_accuracies=True)
        sa50, sa75 = scores[0], scores[5]
        msas.append(msa), sa50s.append(sa50), sa75s.append(sa75)
        print("msa is ", msa)
        print("sa50 is ", sa50)
        print("sa75 is ", sa75)

    return msas, sa50s, sa75s


def run_evaluation(
    gt_paths: List[Union[os.PathLike, str]],
    prediction_paths: List[Union[os.PathLike, str]],
    save_path: Optional[Union[os.PathLike, str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run evaluation for instance segmentation predictions.

    Args:
        gt_folder: The folder with ground-truth images.
        prediction_folder: The folder with the instance segmentations to evaluate.
        save_path: Optional path for saving the results.
        pattern: Optional pattern for selecting the images to evaluate via glob.
            By default all images with ending .tif will be evaluated.
        verbose: Whether to print the progress.

    Returns:
        A DataFrame that contains the evaluation results.
    """
    assert len(gt_paths) == len(prediction_paths)

    # Perform evaluation
    msas, sa50s, sa75s = _run_evaluation(gt_paths, prediction_paths, verbose=verbose)

    results = pd.DataFrame.from_dict({
        "msa": [np.mean(msas)],
        "sa50": [np.mean(sa50s)],
        "sa75": [np.mean(sa75s)],
    })

    if save_path is not None:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        results.to_csv(save_path, index=False)

    return results


if __name__ == "__main__":

    gt_folder = "/projectnb/rfpm/SAM_kontact/fluorescent/groundtruths/"
    prediction_folder = "/projectnb/rfpm/SAM_kontact/fluorescent/segmentations_vit_b_lm_image_encoder/"

    # List all ground truth files and prediction files
    gt_paths = [os.path.join(gt_folder, filename) for filename in os.listdir(gt_folder) if filename.endswith(".tif")]
    prediction_paths = [os.path.join(prediction_folder, filename) for filename in os.listdir(prediction_folder) if filename.endswith(".tif")]

    save_path = "/projectnb/rfpm/SAM_kontact/fluorescent/evaluations_vit_b_lm_image_encoder/evaluation_results.csv"

    run_evaluation(
    gt_paths,
    prediction_paths,
    save_path,
    verbose = True)
