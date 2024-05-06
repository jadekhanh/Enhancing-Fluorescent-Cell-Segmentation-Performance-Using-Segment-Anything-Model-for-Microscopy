import imageio
import torch
import zarr
import napari
import pdb
import os

import torch_em
from torch_em.model import UNETR

import micro_sam
import micro_sam.util as util
from micro_sam.sam_annotator import annotator_2d


def run_annotator_with_finetuned_fluorescent_model():
    """Run the 2d anntator with a custom (finetuned) model.

    Here, we use the model that is produced by `finetuned_deepcell_full_model.py` and apply it
    for an image from the validation set.
    """
    # take the last frame, which is part of the val set, so the model was not directly trained on it
    zarr_file = zarr.open("/Users/khanhaddress/Desktop/torch-em/fluorescent/val/image_1416.zarr")
    # Access the 'raw' array and retrieve its data
    raw_data = zarr_file['raw'][:]
	
    # set the checkpoint and the path for caching the embeddings
    checkpoint = "/Users/khanhaddress/Desktop/torch-em/fluorescent/finetuned sam vit-b fluorescent models/finetuned_deepcell_full_model.pth"
    # checkpoint = "/Users/khanhaddress/Desktop/torch-em/fluorescent/finetuned sam vit-b fluorescent models/finetuned_deepcell_image_encoder_model.pth"
    # checkpoint = "/Users/khanhaddress/Desktop/torch-em/fluorescent/finetuned sam vit-b fluorescent models/finetuned_deepcell_prompt_encoder_model.pth"
    # checkpoint = "/Users/khanhaddress/Desktop/torch-em/fluorescent/finetuned sam vit-b fluorescent models/finetuned_deepcell_mask_decoder_model.pth"
    embedding_path = "/Users/khanhaddress/Desktop/torch-em/fluorescent/embeddings-finetuned.zarr"
    model_type = "vit_b"  # We finetune a vit_b in the example script.
    # Adapt this if you finetune a different model type, e.g. vit_h
    # Load the custom model.
    predictor = util.get_sam_model(model_type=model_type, checkpoint_path=checkpoint)
    
    # Run the 2d annotator with the custom model.
    annotator_2d(
       raw_data, embedding_path=embedding_path, predictor=predictor, precompute_amg_state=True,
    )


if __name__ == "__main__":
    run_annotator_with_finetuned_fluorescent_model()
