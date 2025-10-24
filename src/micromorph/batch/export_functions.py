import pickle
import pandas as pd
import h5py
import numpy as np


def export_analysis_setting(setting, output_path, method='pickle'):
    """
    NOT YET ACTUALLY IMPLEMENTED IN WORKFLOW - work in progress.
    Export the analysis setting dictionary to a file, in either pickle or txt format.
    :param setting: Dictionary containing the analysis settings
    :param output_path: Path to save the file
    :param method: 'pickle', 'txt'or 'both'
    """
    if method == 'pickle':
        with open(output_path+'.pickle', 'wb') as handle:
            pickle.dump(setting, handle, protocol=pickle.HIGHEST_PROTOCOL)
    elif method == 'txt':
        with open(output_path+'.txt', 'w') as file:
            for key, value in setting.items():
                file.write('%s:%s\n' % (key, value))
    elif method == 'both':
        export_analysis_setting(setting, output_path, 'pickle')
        export_analysis_setting(setting, output_path, 'txt')
    return


def export_full_analysis(bacteria_data: list, output_path, method='all', data_type=None):
    # check if data_type is either Bacteria or Bacteria360

    if data_type is None:
        # throw an error and stop
        raise ValueError('data_type must be specified')
    elif data_type != 'Bacteria' and data_type != 'Bacteria360':
        # throw an error and stop
        raise ValueError('data_type must be either Bacteria or Bacteria360')


    if method == 'pickle':
        # pickle method returns the actual list
        with open(output_path+'.pickle', 'wb') as handle:
            pickle.dump(bacteria_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    elif method == 'csv':
        if data_type == 'Bacteria':
            data_dictionary = get_results_dictionary(bacteria_data)
        elif data_type == 'Bacteria360':
            data_dictionary = get_360_results_dictionary(bacteria_data)

        # convert data_dictionary to a dataframe
        df = pd.DataFrame(data_dictionary)
        df.to_csv(output_path+'.csv', index=False)
    elif method == 'hdf5':
        if data_type == 'Bacteria':
            data_dictionary = get_results_dictionary(bacteria_data)
        elif data_type == 'Bacteria360':
            data_dictionary = get_360_results_dictionary(bacteria_data)

        export_bacteria_to_hdf5(output_path+'.hdf5', data_dictionary)
    elif method == 'all':
        export_full_analysis(bacteria_data, output_path, method='hdf5', data_type=data_type)
        export_full_analysis(bacteria_data, output_path, method='pickle', data_type=data_type)
        export_full_analysis(bacteria_data, output_path, method='csv', data_type=data_type)
    return


def get_results_dictionary(bacteria_data: list) -> dict:
    """
    Convenience function which takes a list of Bacteria objects and returns a dictionary with the data,
    which can then be used to save the results with further functions.

    Parameters
    ----------
    bacteria_data: list
        list of Bacteria objects

    Returns
    -------
    data_dictionary: dict
        dictionary with the data from the Bacteria objects

    Keys
    ----
    frame: list
        list of frame numbers, zero if there was only one frame, or no frame information
    centroid: list
        list of centroid coordinates, in pixels
    medial_axis: list
        list of medial axis coordinates, in pixels
    boundary: list
        list of boundary coordinates, in pixels
    width: list
        list of widths of the bacteria, in units
    all_widths: list
        list of all widths found for each bacterium, in units
    length: list
        list of lengths of the bacteria, in units
    area: list
        list of areas of the bacteria, in pixels
    bounding_box: list
        list of bounding box coordinates, in pixels
    orientation: list
        list of orientation of the bacteria, in degrees
    major_axis_length: list
        list of major axis lengths of the bacteria, in units
    minor_axis_length: list
        list of minor axis lengths of the bacteria, in units
    """

    centroid = []
    medial_axis = []
    boundary = []
    width = []
    all_widths = []
    length = []
    area = []
    bounding_box = []
    orientation = []
    major_axis_length = []
    minor_axis_length = []
    medial_axis_extended = []
    frame = []

    for bacteria in bacteria_data:
        try:
            centroid.append(np.array([bacteria.xc, bacteria.yc]))
        except AttributeError:
            centroid.append(np.array([np.nan, np.nan]))
        medial_axis.append(bacteria.medial_axis)
        medial_axis_extended.append(bacteria.medial_axis_extended)
        width.append(bacteria.width)
        all_widths.append(bacteria.all_widths)
        length.append(bacteria.length)
        area.append(bacteria.area)
        bounding_box.append(bacteria.bbox)
        orientation.append(bacteria.orientation)
        major_axis_length.append(bacteria.axis_major_length)
        minor_axis_length.append(bacteria.axis_minor_length)
        frame.append(bacteria.slice)

        try:
            boundary.append(bacteria.boundary)
        except AttributeError:
            boundary.append(np.array([np.nan, np.nan]))

    data_dictionary = {'frame': frame,
                       'centroid': centroid,
                       'medial_axis': medial_axis,
                       'boundary': boundary,
                       'width': width,
                       'all_widths': all_widths,
                       'length': length,
                       'area': area,
                       'bounding_box': bounding_box,
                       'orientation': orientation,
                       'major_axis_length': major_axis_length,
                       'minor_axis_length': minor_axis_length,
                       'medial_axis_extended': medial_axis_extended}

    return data_dictionary


def get_360_results_dictionary(bacteria_data: list) -> dict:
    """
    Convenience function which takes a list of Bacteria360 objects and returns a dictionary with the data.
    This can then be used to save the results with further functions.

    """

    centroid = []
    width_data = []
    width = []
    length = []
    area = []
    bounding_box = []
    frame = []
    n_lines = []
    magnitude = []

    for bacteria in bacteria_data:
        try:
            centroid.append(np.array([bacteria.xc, bacteria.yc]))
        except AttributeError:
            centroid.append(np.array([np.nan, np.nan]))
        try:
            width_data.append(bacteria.width_data)
        except AttributeError:
            width_data.append(np.nan)
        try:
            width.append(bacteria.width)
        except AttributeError:
            width.append(np.nan)
        try:
            length.append(bacteria.length)
        except AttributeError:
            length.append(np.nan)


        try:
            area.append(bacteria.area)
        except AttributeError:
            area.append(np.nan)
        try:
            bounding_box.append(bacteria.bbox)
        except AttributeError:
            bounding_box.append(np.array([np.nan, np.nan]))
        try:
            frame.append(bacteria.slice)
        except AttributeError:
            frame.append(np.nan)
        try:
            n_lines.append(bacteria.n_lines)
        except AttributeError:
            n_lines.append(np.nan)
        try:
            magnitude.append(bacteria.magnitude)
        except AttributeError:
            magnitude.append(np.nan)

    data_dictionary = {'frame': frame,
                       'centroid': centroid,
                       'width_data': width_data,
                       'width': width,
                       'length': length,
                       'area': area,
                       'bounding_box': bounding_box,
                       'n_lines': n_lines,
                       'magnitude': magnitude}

    return data_dictionary


def export_bacteria_to_hdf5(hdf5_path, data_dictionary):
    try:
        with h5py.File(hdf5_path, 'w') as f:
            for key in data_dictionary.keys():
                f.create_group(key)

                for i, data in enumerate(data_dictionary[key]):
                    if data is None:
                        data = np.nan
                    f[key].create_dataset(str(i), data=data)
    except:
        print("Something went wrong...")
        raise ValueError('Error saving data to hdf5 file')
