import numpy as np
import time
import logging
from scipy.signal import savgol_filter

from skimage.measure import profile_line, regionprops
from micromorph.bacteria.utilities import apply_mask_to_image

from micromorph.bacteria.phase_contrast_fitting import fit_phase_contrast_profile, fit_top_hat_profile
from micromorph.bacteria.fluorescence_fitting import fit_ring_profile

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_single_bacterium(args):
    img, mask, i, bact, options = args
    img_processed = apply_mask_to_image(img, mask == bact.label)
    xc = bact.centroid[1]
    yc = bact.centroid[0]

    updated_options = {'magnitude': bact.major_axis_length * 1.5,
                       'area': bact.area,
                       'label': bact.label,
                       'bbox': bact.bbox}

    options.update(updated_options)

    current_bact = Bacteria360(img_processed, (yc, xc), options=options)

    return current_bact

def run_measure360(img: np.array, mask: np.array, options: dict = dict(), pool=None, close_pool=True):
    """
    Runs the measure360 function on a single image or a stack of images.

    Parameters
    ----------
    img : np.array
        Single frame (or stack), either phase contrast or fluorescence.
    mask : np.array
        Labelled mask from the image.
    options : dict, optional
        Dictionary of options to pass to the measure360 function.

    Returns
    -------
    all_bacteria : list
        List of Bacteria360 objects.
    """
    if pool is None:
        logging.info("Making a pool for multithreading - this only happens once!")
        pool = ThreadPoolExecutor()  # Create executor outside a with block
        logging.info("Pool created!")
    else:
        logging.info("Using existing pool for multithreading")

    all_bacteria = []
    if len(img.shape) == 2:
        # Get bacteria properties
        props = regionprops(mask)

        args = [(img, mask, i, bact, options) for i, bact in enumerate(props)]

        futures = [pool.submit(process_single_bacterium, arg) for arg in args]
        with tqdm(total=len(args)) as pbar:
            for future in as_completed(futures):
                result = future.result()
                all_bacteria.append(result)
                pbar.update(1)
    else:
        n_frames = img.shape[0]
        for i in range(n_frames):
            # Get ith image and mask
            img_current = np.copy(img[i])
            mask_current = np.copy(mask[i])

            current_data = run_measure360(img_current, mask_current, options=options, pool=pool, close_pool=False)

            for bact in current_data:
                bact.slice = i

            all_bacteria.extend(current_data)

    if close_pool:
        if pool:
            pool.shutdown(wait=True)

    return all_bacteria


class Bacteria360:
    """
    A class to store all the properties of a single bacterium.
    It requires an image and a mask to be initialised, which are used to calculate the properties.

    Parameters
    ----------
    img : np.array
        The image of the bacterium.
    centroid : tuple
        The centroid of the bacterium.
    options : dict
        Dictionary of options to pass to the measure360 function.

    Attributes
    ----------
    centroid : tuple
        The centroid of the bacterium.
    width_data : np.array
        The width data of the bacterium.
    width : float
        The minimum width of the bacterium.
    length : float
        The maximum width of the bacterium.
    area : float
        The area of the bacterium.
    label : int
        The label of the bacterium.
    bbox : tuple
        The bounding box of the bacterium.
    slice : int
        The slice of the stack the bacterium is from.
    n_lines : int
        The number of lines to measure the width along.
    magnitude : float
        The length of the lines.
    px_size : float
        The pixel size of the image.
    psfFWHM : float
        The FWHM of the PSF.
    xc : float
        The x-coordinate of the centroid.
    yc : float
        The y-coordinate of the centroid.
    xc_nm : float
        The x-coordinate of the centroid in nm.
    yc_nm : float
        The y-coordinate of the centroid in nm.

    """
    def __init__(self, img, centroid, options=dict()):

        n_lines = options.get('n_lines', 50)
        magnitude = options.get('magnitude', 100)
        pool = options.get('pool', None)
        area = options.get('area', 0)
        label = options.get('label', None)
        bbox = options.get('bbox', None)

        width_data = np.array(measure360(img, centroid, options=options))

        # remove any rows containing nan values
        width_data = width_data[~np.isnan(width_data).any(axis=1)]

        self.centroid = centroid
        self.width_data = width_data
        self.width = np.min(self.width_data[:, 1])
        self.length = np.max(self.width_data[:, 1])
        self.area = area
        self.label = label
        self.bbox = bbox
        self.slice = 0  # this is by default 0, but gets overwritten when data is generated from a stack

        self.n_lines = n_lines
        self.magnitude = magnitude
        self.px_size = options.get('px_size', None)
        self.psfFWHM = options.get('psfFWHM', None)

        self.xc = centroid[0]
        self.yc = centroid[1]

        self.xc_nm = self.xc * options.get('px_size', 1)
        self.yc_nm = self.yc * options.get('px_size', 1)


def fit_multiprocessing(img, angle, x_1, y_1, x_2, y_2, params):
    """
    Function to fit the width of the bacterium at a single angle.

    Parameters
    ----------
    img : np.array
        The image of the bacterium.
    angle : float
        The angle corresponding to the line. It is stores in the results array but has no bearing on the width
        calculation.
    x_1 : float
        The x-coordinate of the start of the line.
    y_1 : float
        The y-coordinate of the start of the line.
    x_2 : float
        The x-coordinate of the end of the line.
    y_2 : float
        The y-coordinate of the end of the line.
    params : list
        List of parameters [psfFWHM, pxsize, fit_type].

    Returns
    -------
    data : np.array
        Array of data [angle, width, x_1, y_1, x_2, y_2].
    """

    psfFWHM = params[0]
    pxsize = params[1]
    fit_type = params[2]

    # Set up array to collect data
    current_profile = profile_line(img, (y_1, x_1), (y_2, x_2))

    x = np.arange(0, len(current_profile)) * pxsize

    try:
        if fit_type == 'fluorescence':
            result = fit_ring_profile(x, current_profile, psfFWHM)
            width = result.params['R'] * 2
        elif fit_type == 'phase':
            result = fit_phase_contrast_profile(x, current_profile)
            width = result.params['width'].value
        elif fit_type == 'tophat':
            result = fit_top_hat_profile(x, current_profile)
            width = result.params['width'].value
        else:
            raise ValueError('Invalid fit type - please choose "fluorescence" or "phase".')
    except:
        width = np.nan

    if width < 0:
        width = np.nan

    data = np.array([angle, width, x_1, y_1, x_2, y_2])

    return data


def measure360(img, centroid, options=dict()):
    """
    This function measures the width of a bacterium at 360 degrees around its centroid.

    Parameters
    ----------
    img : np.array
        The image of the bacterium.
    centroid : tuple
        The centroid of the bacterium.
    options : dict
        Dictionary of options to pass to the measure360 function.

    Returns
    -------
    data : list
        List of data [angle, width, x_1, y_1, x_2, y_2].
    """
    n_lines = options.get('n_lines', 100)
    magnitude = options.get('magnitude', 100)
    psfFWHM = options.get('psfFWHM', 0.25)
    px_size = options.get('pxsize', 0.65)
    fit_type = options.get('fit_type', 'fluorescence')
    pool = options.get('pool', None)
    params = [psfFWHM, px_size, fit_type]

    # # Initialise multi processing pool
    # if pool is None:
    #     # Number of processes to run in parallel
    #     num_processes = multiprocessing.cpu_count()
    #     # Create a pool of worker processes
    #     pool = multiprocessing.Pool(processes=num_processes)

    # Define angle_range based on required number of lines
    angle_range = np.linspace(0, np.pi, n_lines)

    # Get points on circle
    x_1, y_1, x_2, y_2 = get_points_on_circle(centroid, magnitude/2, angle_range)

    # Check if any x or y is below 0 or above the image size
    x_1 = np.clip(x_1, 0, img.shape[1])
    y_1 = np.clip(y_1, 0, img.shape[0])
    x_2 = np.clip(x_2, 0, img.shape[1])
    y_2 = np.clip(y_2, 0, img.shape[0])

    data = []

    start = time.time()

    # possible way to run multiprocessing?
#     with multiprocessing.Pool() as pool:
#        data = pool.map(fit_multiprocessing, [(img, angle_range[i], x_1[i], y_1[i], x_2[i], y_2[i], params) for i in range(len(x_1))])
    # data = pool.starmap(fit_multiprocessing, [(img, angle_range[i], x_1[i], y_1[i], x_2[i], y_2[i], params) for i in range(len(x_1))])
        
    for i in range(len(x_1)):
        # logging.debug(f'Processing line {i+1} of {len(x_1)}')
        data.append(fit_multiprocessing(img, angle_range[i], x_1[i], y_1[i], x_2[i], y_2[i], params))

    if options.get('verbose', False):
        logging.debug('Time taken: ', time.time() - start)

    return data


def get_points_on_circle(centroid, magnitude, angle_range):
    """
    Function to get the points on a circle of a given radius around a centroid.

    Parameters
    ----------
    centroid : tuple
        The centroid of the bacterium.
    magnitude : float
        The radius of the circle.
    angle_range : np.array
        The range of angles to calculate the points.

    Returns
    -------
    x_1 : np.array
        The x-coordinates of the start of the lines.
    y_1 : np.array
        The y-coordinates of the start of the lines.
    x_2 : np.array
        The x-coordinates of the end of the lines.
    y_2 : np.array
        The y-coordinates of the end of the lines.
    """
    x_1 = centroid[1] + magnitude*np.cos(angle_range)
    y_1 = centroid[0] + magnitude*np.sin(angle_range)

    x_2 = centroid[1] - magnitude*np.cos(angle_range)
    y_2 = centroid[0] - magnitude*np.sin(angle_range)

    return x_1, y_1, x_2, y_2


def width_distribution(all_data):
    """
    Convenience function to plot the distribution of the widths of all the bacteria.

    Parameters
    ----------
    all_data : list
        List of Bacteria360 objects.

    Returns
    -------
    None
    """

    all_widths = []

    for data in all_data:
        width = data.width
        all_widths.append(width)

    plt.hist(all_widths)
    plt.show()
    print(len(all_widths))


def filter_measure360(bacteria, filter_type, filter_settings):
    """
    Function to filter the width data of a bacterium.

    Parameters
    ----------
    bacteria : Bacteria360
        The Bacteria360 object to be filtered.
    filter_type : str
        The type of filter to apply. Options are 'stdev', 'derivative' or 'sav-gol'.
    filter_settings : list
        The settings for the filter. For 'stdev' and 'derivative' this is a single value, for 'sav-gol' it is a list
        of two values [window, order].

    Returns
    -------
    bacteria : Bacteria360
        The Bacteria360 object with the filtered width data.
    """
    width_data = np.copy(bacteria.width_data[:, 1])

    if filter_type == 'stdev':
        # Filter the width
        idx = abs(width_data - np.median(width_data)) < filter_settings[0] * np.std(width_data)

        bacteria.width_data = bacteria.width_data[idx, :]
    elif filter_type == 'derivative':
        delta = np.diff(width_data)
        delta = np.insert(delta, 0, 0)
        idx = abs(delta) < filter_settings[0]

        bacteria.width_data = bacteria.width_data[idx, :]
    elif filter_type == 'sav-gol':
        window = filter_settings[0]
        order = filter_settings[1]
        width_data = savgol_filter(width_data, window, order)
        bacteria.width_data[:, 1] = width_data

    try:
        bacteria.width = np.min(bacteria.width_data[:, 1])
        bacteria.length = np.max(bacteria.width_data[:, 1])
    except:
        print("filtering didn't work")

    return bacteria

