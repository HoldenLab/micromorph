from skimage.io import imread
import matplotlib.pyplot as plt
import matplotlib as mpl
from omnipose.utils import rescale
import fastremap
import ncolor
import numpy as np

from micromorph import get_bacteria_list
from micromorph.segmentation import run_omnipose
from micromorph.measure360 import run_measure360

if __name__ == "__main__":
    # Load image and the corresponding mask
    image = imread(r"C:\Users\u1870329\Documents\GitHub\micromorph\micromorph\test-data\microcolony_omnisegger_example.tif")

    # We now generate the mask using Omnipose
    mask, _ = run_omnipose(image, 'bact_phase_omni', gpu_option=False, filter_options={'min_area': 50})
    # Now use the `get_bacteria_list` function to analyse all the bacteria in the image
    # bacteria_list = get_bacteria_list(image, mask, options={'pxsize': 110, 'fit_type': 'phase'})
    bacteria_list = run_measure360(image, mask, options={'n_angles': 50, 'pxsize': 65, 'fit_type': 'phase', 'psfFWHM': 250})
