import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# line above is a workaround for this problem : https://stackoverflow.com/questions/20554074/sklearn-omp-error-15-initializing-libiomp5md-dll-but-found-mk2iomp5md-dll-a
from skimage.measure import regionprops
import time
from typing import Union
from cellpose_omni import models
import logging

def run_omnipose(images: np.array, model_name: str, gpu_option: bool, chans: tuple or None =None, filter_options:
dict ={}) -> Union[np.array, np.array]:
    """
    Run the specified omnipose model on the input image(s)

    Parameters
    ----------
    images: np.ndarray
        The input image(s) to segment. If a single image is provided, it should be a 2D array. If multiple images are provided, they should be a 3D array with shape (n_images, height, width).
    model_name: str
        The name of the model to use.
    gpu_option: bool
        Whether to use the GPU for segmentation.
    chans: tuple or None
        The channels to use for segmentation. If None, the first channel will be used.
    filter_options: dict
        A dictionary containing the filter options for the segmentation. Related to the omnipose package. See this 
        link for more info: https://omnipose.readthedocs.io/examples/mono_channel_bact.html#run-segmentation
              
    Returns
    -------
    masks: np.ndarray
        The segmented masks. If a single image is provided, it will be a 2D array. If multiple images are provided, it will be a 3D array with shape (n_images, height, width).
    flows: np.ndarray
        The flow outputs. If a single image is provided, it will be a 2D array. If multiple images are provided, it will be a 3D array with shape (n_images, height, width).
    """
    # Should also add to this function the possibility to:
    model = models.CellposeModel(gpu=gpu_option, model_type=model_name)

    if chans is None:
        chans = [0, 0]  # If user didn't specify a channel, assume you need to use the first channel

    # TODO: params should be a dictionary which you can edit

    # define parameters - this is to become an input parameter
    params = {'channels': chans,  # always define this with the model
              'rescale': None,  # upscale or downscale your images, None = no rescaling
              'mask_threshold': 1,  # erode or dilate masks with higher or lower values -1
              'flow_threshold': 0.2,
              # default is .4, but only needed if there are spurious masks to clean up; slows down output
              'transparency': True,  # transparency in flow output
              'omni': True,  # we can turn off Omnipose mask reconstruction, not advised
              'cluster': True,  # use DBSCAN clustering
              'resample': True,  # whether to run dynamics on rescaled grid or original grid
              # 'verbose': False, # turn on if you want to see more output
              'tile': False,  # average the outputs from flipped (augmented) images; slower, usually not needed
              'niter': None,
              # None lets Omnipose calculate # of Euler iterations (usually <20) but you can tune it for over/under segmentation
              'augment': False,  # Can optionally rotate the image and average outputs, usually not needed
              'affinity_seg': False,  # new feature, stay tuned...
              }

    tic = time.time()
    if len(images.shape) == 2:
        try:
            masks, flows, styles = model.eval(images, **params)
        except ValueError:
            logging.error("Error in Omnipose segmentation. A blank mask will be returned.")
            masks = np.zeros_like(images)
            flows = np.zeros_like(images)
    else:
        n_images = len(images)
        n = range(n_images)

        masks_list, flows_list, styles = model.eval([images[i] for i in n], **params)

        # Convert list of masks to 3D array
        masks = np.dstack(masks_list)  # Transpose to match the original shape of the stack
        masks = np.moveaxis(masks, -1, 0)

        flows = np.copy(masks)

        for i, sublist in enumerate(flows_list):
            flows[i] = sublist[-1]  # -1

    net_time = time.time() - tic
    logging.info('Total segmentation time: {}s'.format(net_time))

    if filter_options:
        masks = filter_mask(masks, filter_options)

    return masks, flows


def filter_mask(mask:np.array, options: dict ={}) -> np.array:
    """
    Function to filter the segmentation mask basaed on basic morphological criteria such as area, length and width.

    Parameters
    ----------
    mask: np.ndarray
        The mask to filter. If a single mask is provided, it should be a 2D array. If multiple masks are provided, they should be a 3D array with shape (n_masks, height, width).
    options: dict
        A dictionary containing the filter options for the segmentation. The options are:
        - min_area: The minimum area of the bacteria.
        - max_area: The maximum area of the bacteria.
        - min_length: The minimum length of the bacteria.
        - max_length: The maximum length of the bacteria.
        - min_width: The minimum width of the bacteria.
        - max_width: The maximum width of the bacteria.

    Returns
    -------
    filtered_mask: np.ndarray
        The filtered mask. If a single mask is provided, it will be a 2D array. If multiple masks are provided, it will be a 3D array with shape (n_masks, height, width).
    """
    # Based on this: https://forum.image.sc/t/need-help-with-regionprops/30107/2

    # Get values from dictionary if they exist, otherwise use default values
    min_area = options.get("min_area", None)
    max_area = options.get("max_area", None)

    min_length = options.get("min_length", None)
    max_length = options.get("max_length", None)

    min_width = options.get("min_width", None)
    max_width = options.get("max_width", None)

    # Not sure if needed ---
    # label_mask = label(mask)
    label_mask = np.copy(mask)

    if len(label_mask.shape) == 2:
        # Start by getting regionprops
        regions = regionprops(label_mask)

        # Create a list to store the indices of the bacteria that meet the criteria
        bacteria_indices = []
        # Create a new mask with only the bacteria that meet the criteria
        filtered_mask = np.zeros_like(mask)

        # Loop through the regions
        for i, region in enumerate(regions):
            # Get the area, length and width of the region
            area = region.area
            length = region.axis_major_length
            width = region.axis_minor_length

            # Check if the region meets the criteria
            if (min_area is None or area >= min_area) and (max_area is None or area <= max_area) and \
                    (min_length is None or length >= min_length) and (max_length is None or length <= max_length) and \
                    (min_width is None or width >= min_width) and (max_width is None or width <= max_width):
                # If the region meets the criteria, add the index to the list
                bacteria_indices.append(region.label)

        for i in bacteria_indices:
            filtered_mask[label_mask == i] = i
    elif len(label_mask.shape) == 3:
        # Create stack to store the filtered masks
        filtered_mask = np.zeros_like(mask)

        # Loop through the frames
        for i, current_mask in enumerate(label_mask):
            current_filtered_mask = filter_mask(current_mask, options=options)

            filtered_mask[i] = current_filtered_mask
    else:
        raise ValueError("Mask must be 2D or 3D.")

    return filtered_mask
