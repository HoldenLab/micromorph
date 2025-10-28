import time
import numpy as np
from skimage.measure import regionprops, label
from skimage.segmentation import clear_border
from .shape_analysis import (get_bacteria_length, get_bacteria_widths, get_bacteria_boundary, get_medial_axis,
                            smooth_medial_axis, extend_medial_axis)
from .utilities import apply_mask_to_image
import logging
from tqdm.contrib.concurrent import process_map

from tqdm import tqdm
from multiprocessing import Pool


class Bacteria:
    """
    A class to store all the properties of a single bacterium.

    Parameters
    ----------
    img: the image containg the bacterium
    mask: the mask of the bacterium
    options: a dictionary with options to be passed to the functions used to calculate the morphological properties of the bacterium

    Attributes
    ----------
    centroid: the centroid of the bacterium
    xc: the x-coordinate of the centroid
    yc: the y-coordinate of the centroid
    xc_nm: the x-coordinate of the centroid in nm
    yc_nm: the y-coordinate of the centroid in nm
    bbox: the bounding box of the bacterium
    axis_major_length: the major axis length of the bacterium
    axis_minor_length: the minor axis length of the bacterium
    orientation: the orientation of the bacterium
    area: the area of the bacterium
    boundary: the boundary of the bacterium
    medial_axis: the medial axis of the bacterium
    medial_axis_extended: the medial axis of the bacterium extended to the boundary
    slice: the slice of the stack where the bacterium is located (default is zero)
    all_widths: the widths of the bacterium at different points along the medial axis
    """

    def __init__(self, img, mask, options=dict()):
        # print("New version!!")
        pxsize = options.get('pxsize', 1)
        n_widths = options.get('n_widths', 5)
        boundary_smoothing_factor = options.get('boundary_smoothing_factor', 8)
        fit_type = options.get('fit_type', None)
        psfFWHM = options.get('psfFWHM', 0.250)
        error_threshold = options.get('error_threshold', 0.05)
        max_iter = options.get('max_iter', 50)
        min_distance_to_boundary = options.get('min_distance_to_boundary', 1)
        step_size = options.get('step_size', 1)

        # TODO: add to GUI!
        spline_spacing = options.get('spline_spacing', 0.25)
        spline_val = options.get('spline_val', 3)


        # Get properties from regionprops, then add them to the class.
        bacteria_props = regionprops(label(mask))

        self.centroid = np.asarray(bacteria_props[0].centroid)
        self.xc = bacteria_props[0].centroid[1]
        self.yc = bacteria_props[0].centroid[0]
        self.xc_nm = self.xc * pxsize
        self.yc_nm = self.yc * pxsize
        self.bbox = bacteria_props[0].bbox
        self.axis_major_length = bacteria_props[0].major_axis_length * pxsize
        self.axis_minor_length = bacteria_props[0].minor_axis_length * pxsize
        self.orientation = bacteria_props[0].orientation
        self.area = bacteria_props[0].area * pxsize**2

        # Apply filter to image to avoid possible issues when calculating width of bacteria in clumps.
        # also, crop image to reduce computational time of erosion etc
        bbox = self.bbox
        mask_cropped = np.copy(mask[bbox[0]-1:bbox[2]+1, bbox[1]-1:bbox[3]+1])
        image_cropped = np.copy(img[bbox[0]-1:bbox[2]+1, bbox[1]-1:bbox[3]+1])

        if fit_type == 'phase':
            # Apply mask to image
            img_filtered = apply_mask_to_image(image_cropped, mask_cropped, method='max')
        else:
            # Apply mask to image
            img_filtered = apply_mask_to_image(image_cropped, mask_cropped, method='min')

        # img_filtered = apply_mask_to_image(image_cropped, mask_cropped)

        # Calculate other properties with custom functions.
        try:
            self.boundary = get_bacteria_boundary(mask_cropped, boundary_smoothing_factor=boundary_smoothing_factor)
            self.perimeter = np.sum(np.sqrt(np.sum(np.diff(self.boundary, axis=0)**2, axis=1))) * pxsize
            self.circularity = (4 * np.pi * self.area) / (self.perimeter**2)

            start_time_medial_ax = time.perf_counter()
            self.medial_axis = smooth_medial_axis(get_medial_axis(mask_cropped), self.boundary,
                                                  error_threshold=error_threshold, max_iter=max_iter,
                                                  spline_val=spline_val, spline_spacing=spline_spacing)
            end_time_medial_ax = time.perf_counter()
            logging.debug(f"Time taken to calculate medial axis: {end_time_medial_ax - start_time_medial_ax:.2f} "
                          f"seconds")

            start_time_extend = time.perf_counter()
            self.medial_axis_extended = extend_medial_axis(self.medial_axis, self.boundary,
                                                           error_threshold=error_threshold, max_iter=max_iter,
                                                           min_distance_to_boundary=min_distance_to_boundary,
                                                           step_size=step_size)
            end_time_extend = time.perf_counter()
            logging.debug(f"Time taken to extend medial axis: {end_time_extend - start_time_extend:.2f} seconds")

            start_time_widths = time.perf_counter()
            logging.debug(f"Calculating widths using method {fit_type}")
            self.all_widths = get_bacteria_widths(img_filtered,
                                                  self.medial_axis,
                                                  n_lines=n_widths, pxsize=pxsize, fit_type=fit_type,
                                                  line_magnitude=bacteria_props[0].axis_minor_length*1.5,
                                                  psfFWHM=psfFWHM)
            end_time_widths = time.perf_counter()
            logging.debug(f"Time taken to calculate widths: {end_time_widths - start_time_widths:.2f} seconds")

            self.width = np.median(self.all_widths)
            if self.width < 0:
                self.width = None
            self.length = get_bacteria_length(self.medial_axis_extended, pxsize)

            # logging.debug(f"Width: {self.width:.2f} nm from {self.all_widths}")
            # logging.debug(f"Length: {self.length:.2f} nm")
            # TODO: improve how errors are dealt with
        except ValueError:
            logging.debug("Error calculating widths or length-VALUE")
            self.medial_axis = None
            self.medial_axis_extended = None
            self.all_widths = None
            self.width = None
            self.length = None
        except IndexError:
            logging.debug("Error calculating widths or length-INDEX")
            self.medial_axis = None
            self.medial_axis_extended = None
            self.all_widths = None
            self.width = None
            self.length = None

        # Set slice to zero, this gets updated if the image is part of a stack outside of this function.
        self.slice = 0


def get_bacteria_for_index(img: np.array, mask: np.array, i: int, options: dict):
    """
    Function which returns a Bacteria object for the i-th bacterium in the labelled mask.

    Parameters
    ----------
    img: np.array
        the image containing all the bacteria
    mask: np.array
        the labelled mask of the image
    i: int
        the index of the bacterium to be analysed
    options: dict
        a dictionary with options to be passed to the functions used to calculate the morphological properties of
        the bacterium

    Returns
    -------
    bac: Bacteria
        a Bacteria object containing the properties of the i-th bacterium
    """
    mask_current = np.copy(mask) == i

    bac = Bacteria(img, mask_current, options)
    bac.label = i

    return bac



def _process_bacterium(args):
    """Convenience function used for multi-processing."""
    try:
        img, mask, i, options = args
        return get_bacteria_for_index(img, mask, i, options)
    except Exception as e:
        logging.error(f"Error processing bacterium {i}: {e}")
        return None


def get_bacteria_list(img: np.array, mask_original: np.array, options: dict, pool=None, close_pool=True) -> list:
    if pool is None:
        logging.info("Making a pool for multiprocessing - this only happens once!")
        pool = Pool()
        logging.info("Pool created!")
    else:
        pass
        # logging.info("Using existing pool for multiprocessing")

    mask = label(mask_original)
    
    all_bacteria = []

    if len(img.shape) == 2:
        mask = clear_border(mask)  # Clear border to avoid issues with edge bacteria
        n_cells = np.max(mask)
        unique_values = np.unique(mask)
        # remove 0
        unique_values = unique_values[unique_values > 0]
        # args = [(img, mask, j, options) for j in range(1, n_cells + 1)]
        args = [(img, mask, j, options) for j in unique_values]

        with tqdm(total=len(args)) as pbar:
            for result in pool.map(_process_bacterium, args):
                all_bacteria.append(result)
                pbar.update(1)

        # Remove None values
        all_bacteria = [x for x in all_bacteria if x is not None]
    else:
        for i in range(img.shape[0]):
            current_img = np.copy(img[i])
            current_mask = clear_border(np.copy(mask[i]))
            current_bacteria = get_bacteria_list(current_img, current_mask, options, pool=pool, close_pool=False)
            for bact in current_bacteria:
                bact.slice = i
            all_bacteria.extend(current_bacteria)

    if close_pool:
        pool.close()
        pool.join()
    return all_bacteria