import napari
import imageio.v3 as imageio
import zarr

from micro_sam.instance_segmentation import (
    load_instance_segmentation_with_decoder_from_checkpoint, mask_data_to_segmentation
)
from micro_sam.util import precompute_image_embeddings


def run_instance_segmentation_with_finetuned_model():
    """Run the instance segmentation with the finetuned model.

    Here, we use the model that is produced by `finetuned_hela.py` and apply it
    for an image from the validation set.
    This only works if 'instance_segmentation_with_decoder' was set to true.
    """
    # take the last frame, which is part of the val set, so the model was not directly trained on it
    zarr_file = zarr.open("/Users/khanhaddress/Desktop/torch-em/fluorescent/test/image_0716.zarr")
    # Access the 'raw' array and retrieve its data
    image = zarr_file['raw'][:]

    # set the checkpoint and the path for caching the embeddings
    checkpoint = "/Users/khanhaddress/Desktop/torch-em/fluorescent/checkpoints/finetuned_prompt_encoder_vit_b_lm/best.pt"
    embedding_path = "/Users/khanhaddress/Desktop/torch-em/fluorescent/embeddings/embeddings-deepcell-vit-b-lm-prompt-encoder.zarr"

    model_type = "vit_b"  # We finetune a vit_b in the example script.
    # Adapt this if you finetune a different model type, e.g. vit_h.

    # Create the segmenter.
    segmenter = load_instance_segmentation_with_decoder_from_checkpoint(checkpoint, model_type=model_type)
    image_embeddings = precompute_image_embeddings(segmenter._predictor, image, embedding_path)

    # Compute the segmentation for the current image:
    # First initialize the segmenter.
    segmenter.initialize(image, image_embeddings)
    # Then compute the actual segmentation. You can set different hyperparameters here,
    # see the function description of 'generate' for details
    masks = segmenter.generate(center_distance_threshold = 0.5,
        boundary_distance_threshold = 0.5,
        foreground_threshold = 0.5,
        distance_smoothing = 0.5,
        min_size = 0)
    segmentation = mask_data_to_segmentation(masks, with_background=True)

    viewer = napari.Viewer()
    viewer.add_image(image)
    viewer.add_labels(segmentation)
    napari.run()


if __name__ == "__main__":
    run_instance_segmentation_with_finetuned_model()
