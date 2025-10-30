# micromorph
![alt text](logo.png)
`micromorph` is a python package designed to perform morphological measurements of bacteria. It is available to use as a python package, or through a graphical interface via a napari plugin.

Link to preprint?

# Installation

If you only want to use the python package, you can install it by running

`pip install micromorph`

If you want to  install the dependencies required to open the `micromorph-batchtools` utility, run

`pip install micromorph[gui]`

If you would like to install the more complete gui available with the napari plugin, please refer to the instructions at [napari-micromorph](https://github.com/HoldenLab/napari-micromorph).

# Documentation

Documentation is available here.

# Example Usage

We provide some example scripts and example data, which you can find at `examples` and `test-data`.

# Contributing

Feel free to open issues or pull requests if you spot any bugs to our software or would like to suggest improvements!

You can install an editable version of the package by cloning this GitHub repository and installing the local version of the package on your environment by running `pip install -e .` after navigating to the folder in your terminal.

Packages required to build the documentation are available if you install the software with `pip install micromorph[docs]`. You can then build the documentation by running the command `mkdocs serve`, which will make the documentation available locally (typically at `http://127.0.0.1:8000/`). You can add the flag `--livereload` to ensure the documentation is automatically rebuilt upon editing. 