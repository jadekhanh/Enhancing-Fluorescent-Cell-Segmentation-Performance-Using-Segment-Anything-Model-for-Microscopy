import os

import torch
import numpy as np

import segment_anything.utils.amg as amg_utils
from segment_anything import SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide

from . import util
from .instance_segmentation import mask_data_to_segmentation
from ._vendored import batched_mask_to_box
from skimage import io
import pickle

from tqdm import tqdm
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union, Tuple

import imageio.v3 as imageio
from skimage.segmentation import relabel_sequential


@torch.no_grad()
def batched_inference(
    predictor: SamPredictor,
    image: np.ndarray,
    batch_size: int,
    boxes: Optional[np.ndarray] = None,
    points: Optional[np.ndarray] = None,
    point_labels: Optional[np.ndarray] = None,
    multimasking: bool = False,
    embedding_path: Optional[Union[str, os.PathLike]] = None,
    return_instance_segmentation: bool = True,
    segmentation_ids: Optional[list] = None,
    reduce_multimasking: bool = True
):
    """Run batched inference for input prompts.

    Args:
        predictor: The segment anything predictor.
        image: The input image.
        batch_size: The batch size to use for inference.
        boxes: The box prompts. Array of shape N_PROMPTS x 4.
            The bounding boxes are represented by [MIN_X, MIN_Y, MAX_X, MAX_Y].
        points: The point prompt coordinates. Array of shape N_PROMPTS x 2.
            The points are represented by [X, Y].
        point_labels: The point prompt labels. Array of shape N_PROMPTS x 1.
            The labels are either 0 (negative prompt) or 1 (positive prompt).
        multimasking: Whether to predict with 3 or 1 mask.
        embedding_path: Cache path for the image embeddings.
        return_instance_segmentation: Whether to return a instance segmentation
            or the individual mask data.
        segmentation_ids: Fixed segmentation ids to assign to the masks
            derived from the prompts.
        reduce_multimasking: Whether to choose the most likely masks with
            highest ious from multimasking

    Returns:
        The predicted segmentation masks.
    """
    if multimasking and (segmentation_ids is not None) and (not return_instance_segmentation):
        raise NotImplementedError

    if (points is None) != (point_labels is None):
        raise ValueError(
            "If you have point prompts both `points` and `point_labels` have to be passed, "
            "but you passed only one of them."
        )

    have_points = points is not None
    have_boxes = boxes is not None
    if (not have_points) and (not have_boxes):
        raise ValueError("Point and/or box prompts have to be passed, you passed neither.")

    if have_points and (len(point_labels) != len(points)):
        raise ValueError(
            "The number of point coordinates and labels does not match: "
            f"{len(point_labels)} != {len(points)}"
        )

    if (have_points and have_boxes) and (len(points) != len(boxes)):
        raise ValueError(
            "The number of point and box prompts does not match: "
            f"{len(points)} != {len(boxes)}"
        )
    n_prompts = boxes.shape[0] if have_boxes else points.shape[0]

    if (segmentation_ids is not None) and (len(segmentation_ids) != n_prompts):
        raise ValueError(
            "The number of segmentation ids and prompts does not match: "
            f"{len(segmentation_ids)} != {n_prompts}"
        )

    # Compute the image embeddings.
    image_embeddings = util.precompute_image_embeddings(predictor, image, embedding_path, ndim=2)
    util.set_precomputed(predictor, image_embeddings)

    # Determine the number of batches.
    n_batches = int(np.ceil(float(n_prompts) / batch_size))

    # Preprocess the prompts.
    device = predictor.device
    transform_function = ResizeLongestSide(1024)
    image_shape = predictor.original_size
    if have_boxes:
        boxes = transform_function.apply_boxes(boxes, image_shape)
        boxes = torch.tensor(boxes, dtype=torch.float32).to(device)
    if have_points:
        points = transform_function.apply_coords(points, image_shape)
        points = torch.tensor(points, dtype=torch.float32).to(device)
        point_labels = torch.tensor(point_labels, dtype=torch.float32).to(device)

    masks = amg_utils.MaskData()
    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_stop = min((batch_idx + 1) * batch_size, n_prompts)

        batch_boxes = boxes[batch_start:batch_stop] if have_boxes else None
        batch_points = points[batch_start:batch_stop] if have_points else None
        batch_labels = point_labels[batch_start:batch_stop] if have_points else None

        batch_masks, batch_ious, _ = predictor.predict_torch(
            point_coords=batch_points, point_labels=batch_labels,
            boxes=batch_boxes, multimask_output=multimasking
        )

        # If we expect to reduce the masks from multimasking and use multi-masking,
        # then we need to select the most likely mask (according to the predicted IOU) here.
        if reduce_multimasking and multimasking:
            _, max_index = batch_ious.max(axis=1)
            batch_masks = torch.cat([batch_masks[i, max_id][None] for i, max_id in enumerate(max_index)]).unsqueeze(1)
            batch_ious = torch.cat([batch_ious[i, max_id][None] for i, max_id in enumerate(max_index)]).unsqueeze(1)

        batch_data = amg_utils.MaskData(masks=batch_masks.flatten(0, 1), iou_preds=batch_ious.flatten(0, 1))
        batch_data["masks"] = (batch_data["masks"] > predictor.model.mask_threshold).type(torch.bool)
        batch_data["boxes"] = batched_mask_to_box(batch_data["masks"])

        masks.cat(batch_data)

    # Mask data to records.
    masks = [
        {
            "segmentation": masks["masks"][idx],
            "area": masks["masks"][idx].sum(),
            "bbox": amg_utils.box_xyxy_to_xywh(masks["boxes"][idx]).tolist(),
            "predicted_iou": masks["iou_preds"][idx].item(),
            "seg_id": idx + 1 if segmentation_ids is None else int(segmentation_ids[idx]),
        }
        for idx in range(len(masks["masks"]))
    ]

    if return_instance_segmentation:
        masks = mask_data_to_segmentation(masks, with_background=False, min_object_size=0)

    return masks

def _run_inference_with_prompts_for_image(
    predictor,
    image,
    gt,
    use_points,
    use_boxes,
    n_positives,
    n_negatives,
    dilation,
    batch_size,
    cached_prompts,
    embedding_path,
):
    gt_ids = np.unique(gt)[1:]
    if cached_prompts is None:
        points, point_labels, boxes = _get_batched_prompts(
            gt, gt_ids, use_points, use_boxes, n_positives, n_negatives, dilation
        )
    else:
        points, point_labels, boxes = cached_prompts

    # Make a copy of the point prompts to return them at the end.
    prompts = deepcopy((points, point_labels, boxes))

    # Use multi-masking only if we have a single positive point without box
    multimasking = False
    if not use_boxes and (n_positives == 1 and n_negatives == 0):
        multimasking = True

    instance_labels = batched_inference(
        predictor, image, batch_size,
        boxes=boxes, points=points, point_labels=point_labels,
        multimasking=multimasking, embedding_path=embedding_path,
        return_instance_segmentation=True,
    )

    return instance_labels, prompts

def run_inference_with_prompts(
    predictor: SamPredictor,
    image_paths: List[Union[str, os.PathLike]],
    gt_paths: List[Union[str, os.PathLike]],
    embedding_dir: Union[str, os.PathLike],
    prediction_dir: Union[str, os.PathLike],
    use_points: bool,
    use_boxes: bool,
    n_positives: int,
    n_negatives: int,
    dilation: int = 5,
    prompt_save_dir: Optional[Union[str, os.PathLike]] = None,
    batch_size: int = 512,
) -> None:
    """Run segment anything inference for multiple images using prompts derived from groundtruth.

    Args:
        predictor: The SegmentAnything predictor.
        image_paths: The image file paths.
        gt_paths: The ground-truth segmentation file paths.
        embedding_dir: The directory where the image embddings will be saved or are already saved.
        use_points: Whether to use point prompts.
        use_boxes: Whether to use box prompts
        n_positives: The number of positive point prompts that will be sampled.
        n_negativess: The number of negative point prompts that will be sampled.
        dilation: The dilation factor for the radius around the ground-truth object
            around which points will not be sampled.
        prompt_save_dir: The directory where point prompts will be saved or are already saved.
            This enables running multiple experiments in a reproducible manner.
        batch_size: The batch size used for batched prediction.
    """
    if not (use_points or use_boxes):
        raise ValueError("You need to use at least one of point or box prompts.")

    if len(image_paths) != len(gt_paths):
        raise ValueError(f"Expect same number of images and gt images, got {len(image_paths)}, {len(gt_paths)}")

    (cached_point_prompts, save_point_prompts, point_prompt_save_path,
     cached_box_prompts, save_box_prompts, box_prompt_save_path) = _get_prompt_caching(
         prompt_save_dir, use_points, use_boxes, n_positives, n_negatives
     )

    os.makedirs(prediction_dir, exist_ok=True)
    for image_path, gt_path in tqdm(
        zip(image_paths, gt_paths), total=len(image_paths), desc="Run inference with prompts"
    ):
        image_name = os.path.basename(image_path)
        label_name = os.path.basename(gt_path)

        # We skip the images that already have been segmented.
        prediction_path = os.path.join(prediction_dir, image_name)
        if os.path.exists(prediction_path):
            continue

        assert os.path.exists(image_path), image_path
        assert os.path.exists(gt_path), gt_path

        im = imageio.imread(image_path)
        gt = imageio.imread(gt_path).astype("uint32")
        gt = relabel_sequential(gt)[0]

        embedding_path = os.path.join(embedding_dir, f"{os.path.splitext(image_name)[0]}.zarr")
        this_prompts, cached_point_prompts, cached_box_prompts = _load_prompts(
            cached_point_prompts, save_point_prompts,
            cached_box_prompts, save_box_prompts,
            label_name
        )
        instances, this_prompts = _run_inference_with_prompts_for_image(
            predictor, im, gt, n_positives=n_positives, n_negatives=n_negatives,
            dilation=dilation, use_points=use_points, use_boxes=use_boxes,
            batch_size=batch_size, cached_prompts=this_prompts,
            embedding_path=embedding_path,
        )

        if save_point_prompts:
            cached_point_prompts[label_name] = this_prompts[:2]
        if save_box_prompts:
            cached_box_prompts[label_name] = this_prompts[-1]

        # It's important to compress here, otherwise the predictions would take up a lot of space.
        imageio.imwrite(prediction_path, instances, compression=5)

    # Save the prompts if we run experiments with prompt caching and have computed them
    # for the first time.
    if save_point_prompts:
        with open(point_prompt_save_path, "wb") as f:
            pickle.dump(cached_point_prompts, f)
    if save_box_prompts:
        with open(box_prompt_save_path, "wb") as f:
            pickle.dump(cached_box_prompts, f)

    # Save the prompts if we run experiments with prompt caching and have computed them
    # for the first time.
    if save_point_prompts:
        with open(point_prompt_save_path, "wb") as f:
            pickle.dump(cached_point_prompts, f)
    if save_box_prompts:
        with open(box_prompt_save_path, "wb") as f:
            pickle.dump(cached_box_prompts, f)


def _get_prompt_caching(prompt_save_dir, use_points, use_boxes, n_positives, n_negatives):

    def get_prompt_type_caching(use_type, save_name):
        if not use_type:
            return None, False, None

        prompt_save_path = os.path.join(prompt_save_dir, save_name)
        if os.path.exists(prompt_save_path):
            print("Using precomputed prompts from", prompt_save_path)
            # We delay loading the prompts, so we only have to load them once they're needed the first time.
            # This avoids loading the prompts (which are in a big pickle file) if all predictions are done already.
            cached_prompts = prompt_save_path
            save_prompts = False
        else:
            print("Saving prompts in", prompt_save_path)
            cached_prompts = {}
            save_prompts = True
        return cached_prompts, save_prompts, prompt_save_path

    # Check if prompt serialization is enabled.
    # If it is then load the prompts if they are already cached and otherwise store them.
    if prompt_save_dir is None:
        print("Prompts are not cached.")
        cached_point_prompts, cached_box_prompts = None, None
        save_point_prompts, save_box_prompts = False, False
        point_prompt_save_path, box_prompt_save_path = None, None
    else:
        cached_point_prompts, save_point_prompts, point_prompt_save_path = get_prompt_type_caching(
            use_points, f"points-p{n_positives}-n{n_negatives}.pkl"
        )
        cached_box_prompts, save_box_prompts, box_prompt_save_path = get_prompt_type_caching(
            use_boxes, "boxes.pkl"
        )

    return (cached_point_prompts, save_point_prompts, point_prompt_save_path,
            cached_box_prompts, save_box_prompts, box_prompt_save_path)

def _load_prompts(
    cached_point_prompts, save_point_prompts,
    cached_box_prompts, save_box_prompts,
    image_name
):

    def load_prompt_type(cached_prompts, save_prompts):
        # Check if we have saved prompts.
        if cached_prompts is None or save_prompts:  # we don't have cached prompts
            return cached_prompts, None

        # we have cached prompts, but they have not been loaded yet
        if isinstance(cached_prompts, str):
            with open(cached_prompts, "rb") as f:
                cached_prompts = pickle.load(f)

        prompts = cached_prompts[image_name]
        return cached_prompts, prompts

    cached_point_prompts, point_prompts = load_prompt_type(cached_point_prompts, save_point_prompts)
    cached_box_prompts, box_prompts = load_prompt_type(cached_box_prompts, save_box_prompts)

    # we don't have anything cached
    if point_prompts is None and box_prompts is None:
        return None, cached_point_prompts, cached_box_prompts

    if point_prompts is None:
        input_point, input_label = [], []
    else:
        input_point, input_label = point_prompts

    if box_prompts is None:
        input_box = []
    else:
        input_box = box_prompts

    prompts = (input_point, input_label, input_box)
    return prompts, cached_point_prompts, cached_box_prompts

def _get_batched_prompts(
    gt,
    gt_ids,
    use_points,
    use_boxes,
    n_positives,
    n_negatives,
    dilation,
):
    # Initialize the prompt generator.
    prompt_generator = PointAndBoxPromptGenerator(
        n_positive_points=n_positives, n_negative_points=n_negatives,
        dilation_strength=dilation, get_point_prompts=use_points,
        get_box_prompts=use_boxes
    )

    # Generate the prompts.
    center_coordinates, bbox_coordinates = util.get_centers_and_bounding_boxes(gt)
    center_coordinates = [center_coordinates[gt_id] for gt_id in gt_ids]
    bbox_coordinates = [bbox_coordinates[gt_id] for gt_id in gt_ids]
    masks = util.segmentation_to_one_hot(gt.astype("int64"), gt_ids)

    points, point_labels, boxes, _ = prompt_generator(
        masks, bbox_coordinates, center_coordinates
    )

    def to_numpy(x):
        if x is None:
            return x
        return x.numpy()

    return to_numpy(points), to_numpy(point_labels), to_numpy(boxes)

class PromptGeneratorBase:
    """PromptGeneratorBase is an interface to implement specific prompt generators.
    """
    def __call__(
            self,
            segmentation: torch.Tensor,
            prediction: Optional[torch.Tensor] = None,
            bbox_coordinates: Optional[List[tuple]] = None,
            center_coordinates: Optional[List[np.ndarray]] = None
    ) -> Tuple[
        Optional[torch.Tensor],  # the point coordinates
        Optional[torch.Tensor],  # the point labels
        Optional[torch.Tensor],  # the bounding boxes
        Optional[torch.Tensor],  # the mask prompts
    ]:
        """Return the point prompts given segmentation masks and optional other inputs.

        Args:
            segmentation: The object masks derived from instance segmentation groundtruth.
                Expects a float tensor of shape NUM_OBJECTS x 1 x H x W.
                The first axis corresponds to the binary object masks.
            prediction: The predicted object masks corresponding to the segmentation.
                Expects the same shape as the segmentation
            bbox_coordinates: Precomputed bounding boxes for the segmentation.
                Expects a list of length NUM_OBJECTS.
            center_coordinates: Precomputed center coordinates for the segmentation.
                Expects a list of length NUM_OBJECTS.

        Returns:
            The point prompt coordinates. Int tensor of shape NUM_OBJECTS x NUM_POINTS x 2.
                The point coordinates are retuned in XY axis order. This means they are reversed compared
                to the standard YX axis order used by numpy.
            The point prompt labels. Int tensor of shape NUM_OBJECTS x NUM_POINTS.
            The box prompts. Int tensor of shape NUM_OBJECTS x 4.
                The box coordinates are retunred as MIN_X, MIN_Y, MAX_X, MAX_Y.
            The mask prompts. Float tensor of shape NUM_OBJECTS x 1 x H' x W'.
                With H' = W'= 256.
        """
        raise NotImplementedError("PromptGeneratorBase is just a class template. \
                                  Use a child class that implements the specific generator instead")


class PointAndBoxPromptGenerator(PromptGeneratorBase):
    """Generate point and/or box prompts from an instance segmentation.

    You can use this class to derive prompts from an instance segmentation, either for
    evaluation purposes or for training Segment Anything on custom data.
    In order to use this generator you need to precompute the bounding boxes and center
    coordiantes of the instance segmentation, using e.g. `util.get_centers_and_bounding_boxes`.

    Here's an example for how to use this class:
    ```python
    # Initialize generator for 1 positive and 4 negative point prompts.
    prompt_generator = PointAndBoxPromptGenerator(1, 4, dilation_strength=8)

    # Precompute the bounding boxes for the given segmentation
    bounding_boxes, _ = util.get_centers_and_bounding_boxes(segmentation)

    # generate point prompts for the objects with ids 1, 2 and 3
    seg_ids = (1, 2, 3)
    object_mask = np.stack([segmentation == seg_id for seg_id in seg_ids])[:, None]
    this_bounding_boxes = [bounding_boxes[seg_id] for seg_id in seg_ids]
    point_coords, point_labels, _, _ = prompt_generator(object_mask, this_bounding_boxes)
    ```

    Args:
        n_positive_points: The number of positive point prompts to generate per mask.
        n_negative_points: The number of negative point prompts to generate per mask.
        dilation_strength: The factor by which the mask is dilated before generating prompts.
        get_point_prompts: Whether to generate point prompts.
        get_box_prompts: Whether to generate box prompts.
    """
    def __init__(
        self,
        n_positive_points: int,
        n_negative_points: int,
        dilation_strength: int,
        get_point_prompts: bool = True,
        get_box_prompts: bool = False
    ) -> None:
        self.n_positive_points = n_positive_points
        self.n_negative_points = n_negative_points
        self.dilation_strength = dilation_strength
        self.get_box_prompts = get_box_prompts
        self.get_point_prompts = get_point_prompts

        if self.get_point_prompts is False and self.get_box_prompts is False:
            raise ValueError("You need to request box prompts, point prompts or both.")

    def _sample_positive_points(self, object_mask, center_coordinates, coord_list, label_list):
        if center_coordinates is not None:
            # getting the center coordinate as the first positive point (OPTIONAL)
            coord_list.append(tuple(map(int, center_coordinates)))  # to get int coords instead of float

            # getting the additional positive points by randomly sampling points
            # from this mask except the center coordinate
            n_positive_remaining = self.n_positive_points - 1

        else:
            # need to sample "self.n_positive_points" number of points
            n_positive_remaining = self.n_positive_points

        if n_positive_remaining > 0:
            object_coordinates = torch.where(object_mask)
            n_coordinates = len(object_coordinates[0])

            # randomly sampling n_positive_remaining_points from these coordinates
            indices = np.random.choice(
                n_coordinates, size=n_positive_remaining,
                # Allow replacing if we can't sample enough coordinates otherwise
                replace=True if n_positive_remaining > n_coordinates else False,
            )
            coord_list.extend([
                [object_coordinates[0][idx], object_coordinates[1][idx]] for idx in indices
            ])

        label_list.extend([1] * self.n_positive_points)
        assert len(coord_list) == len(label_list) == self.n_positive_points
        return coord_list, label_list

    def _sample_negative_points(self, object_mask, bbox_coordinates, coord_list, label_list):
        if self.n_negative_points == 0:
            return coord_list, label_list

        # getting the negative points
        # for this we do the opposite and we set the mask to the bounding box - the object mask
        # we need to dilate the object mask before doing this: we use kornia.morphology.dilation for this
        dilated_object = object_mask[None, None]
        for _ in range(self.dilation_strength):
            dilated_object = morphology.dilation(dilated_object, torch.ones(3, 3), engine="convolution")
        dilated_object = dilated_object.squeeze()

        background_mask = torch.zeros(object_mask.shape, device=object_mask.device)
        _ds = self.dilation_strength
        background_mask[max(bbox_coordinates[0] - _ds, 0): min(bbox_coordinates[2] + _ds, object_mask.shape[-2]),
                        max(bbox_coordinates[1] - _ds, 0): min(bbox_coordinates[3] + _ds, object_mask.shape[-1])] = 1
        background_mask = torch.abs(background_mask - dilated_object)

        # the valid background coordinates
        background_coordinates = torch.where(background_mask)
        n_coordinates = len(background_coordinates[0])

        # randomly sample the negative points from these coordinates
        indices = np.random.choice(
            n_coordinates, replace=False,
            size=min(self.n_negative_points, n_coordinates)  # handles the cases with insufficient bg pixels
        )
        coord_list.extend([
            [background_coordinates[0][idx], background_coordinates[1][idx]] for idx in indices
        ])
        label_list.extend([0] * len(indices))

        return coord_list, label_list

    def _ensure_num_points(self, object_mask, coord_list, label_list):
        num_points = self.n_positive_points + self.n_negative_points

        # fill up to the necessary number of points if we did not sample enough of them
        if len(coord_list) != num_points:
            # to stay consistent, we add random points in the background of an object
            # if there's no neg region around the object - usually happens with small rois
            needed_points = num_points - len(coord_list)
            more_neg_points = torch.where(object_mask == 0)
            indices = np.random.choice(len(more_neg_points[0]), size=needed_points, replace=False)

            coord_list.extend([
                (more_neg_points[0][idx], more_neg_points[1][idx]) for idx in indices
            ])
            label_list.extend([0] * needed_points)

        assert len(coord_list) == len(label_list) == num_points
        return coord_list, label_list

    # Can we batch this properly?
    def _sample_points(self, segmentation, bbox_coordinates, center_coordinates):
        all_coords, all_labels = [], []

        center_coordinates = [None] * len(segmentation) if center_coordinates is None else center_coordinates
        for object_mask, bbox_coords, center_coords in zip(segmentation, bbox_coordinates, center_coordinates):
            coord_list, label_list = [], []
            coord_list, label_list = self._sample_positive_points(
                object_mask[0], center_coords, coord_list, label_list
            )
            coord_list, label_list = self._sample_negative_points(
                object_mask[0], bbox_coords, coord_list, label_list
            )
            coord_list, label_list = self._ensure_num_points(object_mask[0], coord_list, label_list)

            all_coords.append(coord_list)
            all_labels.append(label_list)

        return all_coords, all_labels

    def __call__(
        self,
        segmentation: torch.Tensor,
        bbox_coordinates: List[Tuple],
        center_coordinates: Optional[List[np.ndarray]] = None,
        **kwargs,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        None
    ]:
        """Generate the prompts for one object in the segmentation.

        Args:
            The groundtruth segmentation. Expects a float tensor of shape NUM_OBJECTS x 1 x H x W.
            bbox_coordinates: The precomputed bounding boxes of particular object in the segmentation.
            center_coordinates: The precomputed center coordinates of particular object in the segmentation.
                If passed, these coordinates will be used as the first positive point prompt.
                If not passed a random point from within the object mask will be used.

        Returns:
            Coordinates of point prompts. Returns None, if get_point_prompts is false.
            Point prompt labels. Returns None, if get_point_prompts is false.
            Bounding box prompts. Returns None, if get_box_prompts is false.
        """
        if self.get_point_prompts:
            coord_list, label_list = self._sample_points(segmentation, bbox_coordinates, center_coordinates)
            # change the axis convention of the point coordinates to match the expected coordinate order of SAM
            coord_list = np.array(coord_list)[:, :, ::-1].copy()
            coord_list = torch.from_numpy(coord_list)
            label_list = torch.tensor(label_list)
        else:
            coord_list, label_list = None, None

        if self.get_box_prompts:
            # change the axis convention of the box coordinates to match the expected coordinate order of SAM
            bbox_list = np.array(bbox_coordinates)[:, [1, 0, 3, 2]]
            bbox_list = torch.from_numpy(bbox_list)
        else:
            bbox_list = None

        return coord_list, label_list, bbox_list, None


class IterativePromptGenerator(PromptGeneratorBase):
    """Generate point prompts from an instance segmentation iteratively.
    """
    def _get_positive_points(self, pos_region, overlap_region):
        positive_locations = [torch.where(pos_reg) for pos_reg in pos_region]
        # we may have objects without a positive region (= missing true foreground)
        # in this case we just sample a point where the model was already correct
        positive_locations = [
            torch.where(ovlp_reg) if len(pos_loc[0]) == 0 else pos_loc
            for pos_loc, ovlp_reg in zip(positive_locations, overlap_region)
        ]
        # we sample one location for each object in the batch
        sampled_indices = [np.random.choice(len(pos_loc[0])) for pos_loc in positive_locations]
        # get the corresponding coordinates (Note that we flip the axis order here due to the expected order of SAM)
        pos_coordinates = [
            [pos_loc[-1][idx], pos_loc[-2][idx]] for pos_loc, idx in zip(positive_locations, sampled_indices)
        ]

        # make sure that we still have the correct batch size
        assert len(pos_coordinates) == pos_region.shape[0]
        pos_labels = [1] * len(pos_coordinates)

        return pos_coordinates, pos_labels

    # TODO get rid of this looped implementation and use proper batched computation instead
    def _get_negative_points(self, negative_region_batched, true_object_batched):
        device = negative_region_batched.device

        negative_coordinates, negative_labels = [], []
        for neg_region, true_object in zip(negative_region_batched, true_object_batched):

            tmp_neg_loc = torch.where(neg_region)
            if torch.stack(tmp_neg_loc).shape[-1] == 0:
                tmp_true_loc = torch.where(true_object)
                x_coords, y_coords = tmp_true_loc[1], tmp_true_loc[2]
                bbox = torch.stack([torch.min(x_coords), torch.min(y_coords),
                                    torch.max(x_coords) + 1, torch.max(y_coords) + 1])
                bbox_mask = torch.zeros_like(true_object).squeeze(0)

                custom_df = 3  # custom dilation factor to perform dilation by expanding the pixels of bbox
                bbox_mask[max(bbox[0] - custom_df, 0): min(bbox[2] + custom_df, true_object.shape[-2]),
                          max(bbox[1] - custom_df, 0): min(bbox[3] + custom_df, true_object.shape[-1])] = 1
                bbox_mask = bbox_mask[None].to(device)

                background_mask = torch.abs(bbox_mask - true_object)
                tmp_neg_loc = torch.where(background_mask)

                # there is a chance that the object is small to not return a decent-sized bounding box
                # hence we might not find points sometimes there as well, hence we sample points from true background
                if torch.stack(tmp_neg_loc).shape[-1] == 0:
                    tmp_neg_loc = torch.where(true_object == 0)

            neg_index = np.random.choice(len(tmp_neg_loc[1]))
            neg_coordinates = [tmp_neg_loc[1][neg_index], tmp_neg_loc[2][neg_index]]
            neg_coordinates = neg_coordinates[::-1]
            neg_labels = 0

            negative_coordinates.append(neg_coordinates)
            negative_labels.append(neg_labels)

        return negative_coordinates, negative_labels

    def __call__(
        self,
        segmentation: torch.Tensor,
        prediction: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, None, None]:
        """Generate the prompts for each object iteratively in the segmentation.

        Args:
            The groundtruth segmentation. Expects a float tensor of shape NUM_OBJECTS x 1 x H x W.
            The predicted objects. Epects a float tensor of the same shape as the segmentation.

        Returns:
            The updated point prompt coordinates.
            The updated point prompt labels.
        """
        assert segmentation.shape == prediction.shape
        device = prediction.device

        true_object = segmentation.to(device)
        expected_diff = (prediction - true_object)
        neg_region = (expected_diff == 1).to(torch.float32)
        pos_region = (expected_diff == -1)
        overlap_region = torch.logical_and(prediction == 1, true_object == 1).to(torch.float32)

        pos_coordinates, pos_labels = self._get_positive_points(pos_region, overlap_region)
        neg_coordinates, neg_labels = self._get_negative_points(neg_region, true_object)
        assert len(pos_coordinates) == len(pos_labels) == len(neg_coordinates) == len(neg_labels)

        pos_coordinates = torch.tensor(pos_coordinates)[:, None]
        neg_coordinates = torch.tensor(neg_coordinates)[:, None]
        pos_labels, neg_labels = torch.tensor(pos_labels)[:, None], torch.tensor(neg_labels)[:, None]

        net_coords = torch.cat([pos_coordinates, neg_coordinates], dim=1)
        net_labels = torch.cat([pos_labels, neg_labels], dim=1)

        return net_coords, net_labels, None, None