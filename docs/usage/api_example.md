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
from napari_bacteria_morphology_toolkit._plugin._Bacteria.Bacteria import Bacteria
import matplotlib.pyplot as plt

# Load image and the corresponding mask
image_stack = imread(r"test-data\multiple-cells\example_stack-2.tif")
mask_stack = imread(r"test-data\multiple-cells\example_stack-2_mask_filtered.tif")

# Select the first image and mask
image = image_stack[0]
mask = mask_stack[0] == 1

plotting = False # Set to True to plot the image and mask

if plotting:
    plt.imshow(image)
    plt.imshow(mask)
    plt.show()
else:
    pass

bacteria = Bacteria(image, mask, options={'pxsize': 65, 'n_widths': 5, 'fit_type': 'fluorescence', 'psfFWHM': 250})

print(f"Centroid: {bacteria.centroid}, width: {bacteria.width}, length: {bacteria.length}")
```