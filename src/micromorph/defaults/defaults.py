import json
from pathlib import Path


def get_default() -> dict:
    """
    :return: default options as a dictionary
    """
    filepath = Path(__file__).parent / "default_options.json"
    with open(filepath) as f:
        return json.load(f)


def analysis_options() -> dict:
    """
    Get the default analysis options.
    """
    default_options = get_default()
    analysis_options_dict = default_options['analysis_options']
    return analysis_options_dict


def export_options() -> dict:
    """
    Get the default export options.
    """
    default_options = get_default()
    export_options_dict = default_options['export_options']
    return export_options_dict


def segmentation_options() -> dict:
    """
    Get the default segmentation options.
    """
    default_options = get_default()
    segmentation_options_dict = default_options['segmentation_options']

    return segmentation_options_dict


def image_loading_options() -> dict:
    """
    Get the default image loading options.
    """
    default_options = get_default()
    image_loading_options_dict = default_options['image_loading_options']
    return image_loading_options_dict


def filter_options() -> dict:
    """
    Get the default filter options.
    """
    default_options = get_default()
    filter_options_dict = default_options['filter_options']
    return filter_options_dict


if __name__ == "__main__":
    path = Path(__file__).parent / "default_options.json"
    print(path)
    # print(segmentation_options())
