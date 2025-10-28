# Using the API in a custom script
!!! warning
    This page is still work in progress, so some parts may be incomplete or missing. We are working on it!

The package allows users to directly access the functions and classes used in the napari plugin, meaning that those 
can be easily integrated into custom routines.

Details of the API functions and classes can be found in the API documentation (see menu on the left). 

In this simple example, we load an image and a pre-made mask, and then calculate the properties of the bacteria in 
that image.

```python
from skimage.io import imread
from micromorph import get_bacteria_list
from micromorph.measure360 import run_measure360

import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Load image and the corresponding mask
    image_stack = imread("path/to/image")
    mask_stack = imread("path/to/mask")

    
    # Get measure360 values for the cells in the stack
    bacteria_list = run_measure360(image_stack, mask_stack, options={'n_angles': 50, 'pxsize': 65, 'fit_type': 'phase', 'psfFWHM': 250})

    # or use get_bacteria_list to use "normal mode"
    # bacteria_list = get_bacteria_list(image_stack, mask_stack, options={'pxsize': 65, 'n_widths': 5, 'fit_type': 'fluorescence', 'psfFWHM': 250})

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
```

You can find various example scripts in the [micromorph GitHub repository](https://github.com/HoldenLab/micromorph/tree/master/examples).