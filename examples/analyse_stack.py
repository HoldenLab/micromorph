from skimage.io import imread
from micromorph import get_bacteria_list
from micromorph.measure360 import run_measure360

import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Load image and the corresponding mask
    image_stack = imread(r"C:\Users\u1870329\Documents\GitHub\micromorph\micromorph\test-data\multiple-cells\example_stack.tif")
    mask_stack = imread(r"C:\Users\u1870329\Documents\GitHub\micromorph\micromorph\test-data\multiple-cells\example_stack_mask.tif")
    
    # Get measure360 values for the cells in the stack
    # bacteria_list = run_measure360(image_stack, mask_stack, options={'n_angles': 50, 'pxsize': 65, 'fit_type': 'phase', 'psfFWHM': 250})

    # or use get_bacteria_list to use "normal mode"
    bacteria_list = get_bacteria_list(image_stack, mask_stack, options={'pxsize': 65, 'n_widths': 5, 'fit_type': 'fluorescence', 'psfFWHM': 250})

    # Get widths and lengths
    widths = [bacteria.width for bacteria in bacteria_list]
    lengths = [bacteria.length for bacteria in bacteria_list]

    # Plot histograms
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].hist(widths, bins=30, color='blue', alpha=0.7)
    ax[0].set_title('Width Distribution')

    ax[1].hist(lengths, bins=30, color='green', alpha=0.7)
    ax[1].set_title('Length Distribution')
    plt.show()