from os.path import join
from skimage.io import imsave

from micromorph.segmentation import run_omnipose
from micromorph.batch.batch_utilities import get_tif_list
from micromorph.bacteria import get_bacteria_list
from micromorph.measure360.measure360 import run_measure360, filter_measure360
from micromorph.batch.image_loading_utils import load_image, apply_optosplit
from micromorph.batch.export_functions import export_full_analysis


# Functions for scripting batch processing of data in a folder

def run_omnipose_on_folder(folder_name: str, model_name: str, gpu_option: bool = False, chans=None, verbose=False,
                           mask_filter_options=None, loading_method: str = 'PIL', split_axis=None, split_channel=1):
    """
    Run omnipose on all tif files in a folder.
    :param folder_name: folder containing the tif files
    :param model_name: name of the omnipose model to use
    :param gpu_option: whether omnipose should use the GPU or not
    :param chans: 
    :param verbose: whether to print progress or not
    :param mask_filter_options: options for filtering the masks
    :param loading_method: method to use for loading the images, default is PIL
    :param split_axis: axis to split the image on - if using optosplit
    :param split_channel: channel to split the image on, if using optosplit
    """
    # Get the name of all the tif files in the folder
    file_list = get_tif_list(folder_name)

    # Remove masks from the list
    file_list = [x for x in file_list if 'mask' not in x]

    if verbose:
        print('Found', len(file_list), 'files in folder ', folder_name)

    # Run omnipose on each file
    for file in file_list:
        img = load_image(join(folder_name, file), method=loading_method, verbose=False)

        masks, flows = run_omnipose(img, model_name, gpu_option, filter_options=mask_filter_options)

        save_path = folder_name + '/' + file.replace('.tif', '_mask.tif')

        if split_axis is not None:
            masks = apply_optosplit(masks, optosplit_axis=split_axis, optosplit_channel=split_channel)
            img = apply_optosplit(img, optosplit_axis=split_axis, optosplit_channel=split_channel)

        if len(img.shape) >= 3:
            imsave(save_path, masks, shape=(masks.shape[0],masks.shape[1], masks.shape[2]))
        else:
            imsave(save_path, masks)

        if verbose:
            print('Processed ', file)

    return


def run_analysis_on_folder(folder_name, options=dict()):
    """
    Run the analysis on all tif files in a folder.
    :param folder_name: folder containing the tif files
    :param options: dictionary of options for the analysis
    :return: list of bacteria data
    """
    # Load from options, or set defaults.
    save_flag = options.get('save', True)
    output_type = options.get('output_type', 'all')

    save_directory = folder_name

    loading_method = options.get('loading_method', 'PIL')
    split_axis = options.get('split_axis', None)
    split_channel = options.get('split_channel', 1)

    file_list = get_tif_list(folder_name)

    # Remove masks from the list
    file_list = [x for x in file_list if 'mask' not in x]

    bacteria_data = None

    data_full_folder = []

    for file in file_list:
        img = load_image(join(folder_name, file), method=loading_method, verbose=False,
                         optosplit_channel=split_channel, optosplit_axis=split_axis)

        mask = load_image(join(folder_name, file.replace('.tif', '_mask.tif')), method=loading_method, verbose=False,
                         optosplit_channel=split_channel, optosplit_axis=split_axis)

        output_name = join(save_directory, file.replace('.tif', '_analysis'))

        # Run the analysis
        bacteria_data = get_bacteria_list(img, mask, options=options)

        data_full_folder.extend(bacteria_data)

        if save_flag:
            # Save the analysis for each image
            export_full_analysis(bacteria_data, output_name, output_type, data_type='Bacteria')

    if save_flag:
        # Save the full analysis for the folder
        export_full_analysis(data_full_folder, join(save_directory, 'full_folder_analysis'), output_type, data_type='Bacteria')

    return bacteria_data


def run_measure360_on_folder(folder_name,
                             analysis_options=dict(),
                             export_options=dict(),
                             image_loading_options=dict(),
                             filter_options=dict()):

    # combine the other dictionaries
    options = analysis_options.copy()
    options.update(export_options)
    options.update(image_loading_options)
    options.update(filter_options)

    save_directory = analysis_options.get('save_directory', folder_name)

    save_flag = analysis_options.get('save', True)

    apply_filter = filter_options.get('apply_filter', True)
    filter_type = filter_options.get('filter_type', 'derivative')
    filter_settings = filter_options.get('filter_settings', [250])

    loading_method = image_loading_options.get('loading_method', 'PIL')
    split_axis = image_loading_options.get('split_axis', None)
    split_channel = image_loading_options.get('split_channel', 1)

    output_type = export_options.get('output_type', 'all')

    file_list = get_tif_list(folder_name)
    # Remove masks from the list
    file_list = [x for x in file_list if 'mask' not in x]

    data_full_folder = []

    for file in file_list:
        img = load_image(join(folder_name, file), method=loading_method, verbose=False,
                         optosplit_channel=split_channel, optosplit_axis=split_axis)

        mask = load_image(join(folder_name, file.replace('.tif', '_mask.tif')), method=loading_method, verbose=False,
                          optosplit_channel=split_channel, optosplit_axis=split_axis)

        output_name = join(save_directory, file.replace('.tif', '_measure360'))

        print("Analysing image: ", file)
        bacteria_data = run_measure360(img, mask, options)

        if apply_filter:
            for bacteria in bacteria_data:
                bacteria = filter_measure360(bacteria, filter_type, filter_settings)

        data_full_folder.append(bacteria_data)

        if save_flag:
            export_full_analysis(bacteria_data, output_name, output_type, data_type='Bacteria360')

    if save_flag:
        # Save the full analysis for the folder
        export_full_analysis(data_full_folder, join(save_directory, 'full_folder_measure360'), output_type,
                             data_type='Bacteria360')

    return
