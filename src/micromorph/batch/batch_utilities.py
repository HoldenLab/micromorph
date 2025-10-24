def get_tif_list(folder_path):
    """
    Get a list of tiff files in a folder
    :param folder_path: string, path to the folder
    :return: list of strings, list of tiff files in the folder
    """
    import os
    tiff_list = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    return tiff_list