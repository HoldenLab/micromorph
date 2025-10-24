import numpy as np
from tifffile import imread
import skimage.io as io

def apply_optosplit(img, optosplit_axis=0, optosplit_channel=1,verbose=False):
    img_split = np.zeros(img.shape, dtype=img.dtype)
    if verbose:
        print('Splitting image')

    # check if the image is a stack
    if len(img.shape) == 3:
        if optosplit_channel == 1:
            if optosplit_axis == 0:
                img_split[:, 0:img.shape[0] // 2, :] = img[:, 0:img.shape[0] // 2, :]
            elif optosplit_axis == 1:
                img_split[:, :, 0:img.shape[1] // 2] = img[:, :, 0:img.shape[1] // 2]
        else:
            if optosplit_axis == 0:
                img_split[:, img.shape[0] // 2:, :] = img[:, img.shape[0] // 2:, :]
            elif optosplit_axis == 1:
                img_split[:, :, img.shape[1] // 2:] = img[:, :, img.shape[1] // 2:]
    else:
        if optosplit_channel == 1:
            # Only return half of the columns
            if optosplit_axis == 0:
                img_split[:, 0:img.shape[0] // 2] = img[:, 0:img.shape[0] // 2]
            elif optosplit_axis == 1:
                img_split[0:img.shape[1] // 2, :] = img[0:img.shape[1] // 2, :]
        else:
            if optosplit_axis == 0:
                img_split[:, img.shape[0] // 2:] = img[:, img.shape[0] // 2:]
            elif optosplit_axis == 1:
                img_split[img.shape[1] // 2:, :] = img[img.shape[1] // 2:, :]
    if verbose:
        print('Image split successfully')

    return img_split


def load_image(file_path, method='tifffile', optosplit_channel=1, optosplit_axis=None, verbose=False):
    if method == 'PIL':
        if verbose:
            print('Reading image using PIL')
        try:
            img = io.imread(file_path, plugin='pil')
            if verbose:
                print('Image read successfully')
        except:
            if verbose:
                print('Failed to read image using PIL. Returning None.')
            img = None
    elif method == 'tifffile':
        try:
            if verbose:
                print('Reading image using tifffile')
            img = imread(file_path)
        except:
            print('Could not read file.')
            print('Attempting to use alternative method...')
            try:
                img = io.imread(file_path, plugin='pil')
                print('Alternative method seems to have worked.')
            except:
                print('Failed again. Returning None.')
                img = None
    else:
        print('Method not recognised. Returning None.')
        img = None

    # Additional options can be added here.
    if optosplit_axis is not None:
        img_split = apply_optosplit(img, optosplit_axis=optosplit_axis, optosplit_channel=optosplit_channel, verbose=verbose)

        return img_split
    else:
        return img

