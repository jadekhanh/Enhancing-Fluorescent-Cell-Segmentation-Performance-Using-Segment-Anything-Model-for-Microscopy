import os
import numpy as np
import torch
import glob

from torch_em.model import UNETR
from torch_em.loss import DiceBasedDistanceLoss
from torch_em.transform.label import PerObjectDistanceTransform
from torch_em.transform.raw import normalize_percentile, normalize
from torch_em.transform.label import label_consecutive
from torch_em.data import MinInstanceSampler, ConcatDataset
from torch_em.util.debug import check_loader
from torch_em.data.datasets.dynamicnuclearnet import get_dynamicnuclearnet_loader

import micro_sam.training as sam_training
from micro_sam.training.util import ResizeRawTrafo, ResizeLabelTrafo
from micro_sam.util import export_custom_sam_model

from math import ceil, floor
from typing import Optional, List
from skimage import measure

        
def get_dataloader(split_choice, patch_shape, batch_size, train_instance_segmentation):
    """Return train, val, test dataloader for DeepCELL's DynamicNuclearNet dataset.
    """

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

    loader = get_dynamicnuclearnet_loader(
        path="/projectnb/rfpm/SAM_kontact/fluorescent/datasets/",
        split=split_choice,	# supported for val and test
        patch_shape=(512, 512),
        batch_size=2, 		# 1 if training vit-h, 2 if training vit-b
        sampler=sampler,
        raw_transform=sam_training.identity,
        label_transform=label_transform, download=False
    )

    print("Complete", split_choice, "loader!")
  
    return loader
    
def run_training(checkpoint_name, model_type, train_instance_segmentation):
    """Run the actual model training."""

    # All hyperparameters for training.
    batch_size = 2  # 1 if the training batch size for vit-h, 2 if vit-b
    patch_shape = (512, 512)  # the size of patches for training
    n_objects_per_batch = 5  # the number of objects per batch that will be sampled
    device = torch.device("cuda")  # the device/GPU used for training
    n_iterations = 10000  # how long we train (in iterations)

    # Get the dataloaders.
    train_loader = get_dataloader("train", patch_shape, batch_size, train_instance_segmentation)
    val_loader = get_dataloader("val", patch_shape, batch_size, train_instance_segmentation)

    # Override this to freeze one or more of the backbones
    # freeze_parts = None  # Finetune full model
    # freeze_parts = ["prompt_encoder", "mask_decoder"]   # Finetune image encoder
    freeze_parts = ["image_encoder", "mask_decoder"]    # Finetune prompt encoder
    # freeze_parts = ["image_encoder", "prompt_encoder"]  # Finetune mask decoder

    # Get the segment anything model
    model = sam_training.get_trainable_sam_model(model_type=model_type, device=device, freeze=freeze_parts)
    mem_params = np.sum([param.nelement()*param.element_size() for param in model.parameters()])
    model.to(device)

    # This class creates all the training data for a batch (inputs, prompts and labels).
    convert_inputs = sam_training.ConvertToSamInputs(transform=model.transform, box_distortion_factor=0.025)

    # Get the optimizer and the LR scheduler
    if train_instance_segmentation:
        # for instance segmentation, we use the UNETR model configuration.
        unetr = UNETR(
            backbone="sam", encoder=model.sam.image_encoder, out_channels=3, use_sam_stats=True,
            final_activation="Sigmoid", use_skip_connection=False, resize_input=True,
        )
        mem_params = np.sum([param.nelement()*param.element_size() for param in unetr.parameters()])
        
        # let's get the parameters for SAM and the decoder from UNETR
        joint_model_params = [params for params in model.parameters()]  # sam parameters
        for name, params in unetr.named_parameters():  # unetr's decoder parameters
            if not name.startswith("encoder"):
                joint_model_params.append(params)
        unetr.to(device)
        
        optimizer = torch.optim.Adam(joint_model_params, lr=1e-5)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=10, verbose=True)

    # the trainer which performs training and validation (implemented using "torch_em")
    if train_instance_segmentation:
        instance_seg_loss = DiceBasedDistanceLoss(mask_distances_in_bg=True)
        trainer = sam_training.JointSamTrainer(
            name=checkpoint_name, train_loader=train_loader, val_loader=val_loader, model=model,
            optimizer=optimizer, device=device, lr_scheduler=scheduler, logger=sam_training.JointSamLogger,
            log_image_interval=100, mixed_precision=True, convert_inputs=convert_inputs,
            n_objects_per_batch=n_objects_per_batch, n_sub_iteration=8, compile_model=False, unetr=unetr,
            instance_loss=instance_seg_loss, instance_metric=instance_seg_loss
        )
    else:
        trainer = sam_training.SamTrainer(
            name=checkpoint_name, train_loader=train_loader, val_loader=val_loader, model=model,
            optimizer=optimizer, device=device, lr_scheduler=scheduler, logger=sam_training.SamLogger,
            log_image_interval=100, mixed_precision=True, convert_inputs=convert_inputs,
            n_objects_per_batch=n_objects_per_batch, n_sub_iteration=8, compile_model=False
        )
    trainer.fit(n_iterations)


def export_model(checkpoint_name, model_type):
    """Export the trained model."""
    # export the model after training so that it can be used by the rest of the micro_sam library
    current_directory = os.getcwd()
    export_path = os.path.join(current_directory, "finetuned vit-b-lm models", "finetuned_vit_b_lm_prompt_encoder.pth")
    checkpoint_path = os.path.join(current_directory, "checkpoints", checkpoint_name, "best.pt") # always set to best.pt
    print("current directory", current_directory)
    print("export path", export_path)
    print("checkpoint path", checkpoint_path)
    export_custom_sam_model(
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        save_path=export_path,
    )


def main():
    """
    Finetune a Segment Anything Model.
    """
    torch.cuda.empty_cache()	# clear the cache
    num_gpus = torch.cuda.device_count()
    
    # Iterate over each GPU and print its properties
    for i in range(num_gpus):
        gpu_properties = torch.cuda.get_device_properties(i)
        print(f"GPU {i}:")
        print("  Name:", gpu_properties.name)
        
    # The model_type determines which base model is used to initialize the weights that are finetuned.
    model_type = "vit_b_lm"	# make sure to set the model type

    # The name of the checkpoint. The checkpoints will be stored in './checkpoints/<checkpoint_name>'
    checkpoint_name = "finetuned_prompt_encoder_vit_b_lm"

    # Train an additional convolutional decoder for end-to-end automatic instance segmentation
    train_instance_segmentation = True

    run_training(checkpoint_name, model_type, train_instance_segmentation)
    export_model(checkpoint_name, model_type)


if __name__ =