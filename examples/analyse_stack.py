from skimage.io import imread
from micromorph import get_bacteria_list
import matplotlib.pyplot as plt

# enable logging
import logging
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":

    # Load image and the corresponding mask
    image_stack = imread(r"C:\Users\u1870329\Documents\GitHub\micromorph\micromorph\test-data\multiple-cells\example_stack.tif")
    mask_stack = imread(r"C:\Users\u1870329\Documents\GitHub\micromorph\micromorph\test-data\multiple-cells\example_stack_mask.tif")

    # Print shapes
    print(f"Image stack shape: {image_stack.shape}")
    print(f"Mask stack shape: {mask_stack.shape}")

    bacteria_list = get_bacteria_list(image_stack, mask_stack, options={'pxsize': 65, 'n_widths': 5, 'fit_type': 'fluorescence', 'psfFWHM': 250})

    # # Select the first image and mask
    # image = image_stack[0]
    # mask = mask_stack[0] == 1

    # plotting = False # Set to True to plot the image and mask

    # if plotting:
    #     plt.imshow(image)
    #     plt.imshow(mask)
    #     plt.show()
    # else:
    #     pass

    # bacteria = Bacteria(image, mask, options={'pxsize': 65, 'n_widths': 5, 'fit_type': 'fluorescence', 'psfFWHM': 250})

    for bacteria in bacteria_list:
        print(f"Centroid: ({bacteria.xc:.2f}, {bacteria.yc:.2f}), width: {bacteria.width:.2f}, length: {bacteria.length:.2f}")
