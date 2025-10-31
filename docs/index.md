# micromorph
![alt text](resources/logo-1.png)
`micromorph` is a python package designed to perform morphological measurements of bacterial microscopy images acquired using either phase contrast, or fluorescence microscopy. 

`micromorph` can be used as a package, or through the napari plugin `napari-micromorph`. We recommend starting with the napari plugin to get a sense of what value can be calculated and what parameters are optimal for boundary and midline detection for a specific sample.

You can read about the package and what it can do in our pre-print.

!!! warning
    The documentation is still in development, so some parts may be incomplete or missing. We are 
    working on it!

    
## Installation

You can install the package by running:

```pip install micromorph```

If you also want to install the components necessary to run the batch processing GUI, you can run:

```pip install micromorph[gui]```

If you want to use the napari plugin, you will need to run:

```pip install napari-micromorph```

You will need to have set up a python environment and installed napari before you can make use of most of the package functionality. See [this guide](https://napari.org/dev/tutorials/fundamentals/installation.html) if you are unsure how to do that.

You can also download the plugin through the napari-hub directly inside your napari install.

## Examples

Read the pages in the Usage section, or check out the scripts and notebooks available in the [GitHub repository](https://github.com/HoldenLab/micromorph/tree/master/examples).

## Contributing

Feel free to open issues or pull requests if you spot any bugs to our software or would like to suggest improvements!

You can install an editable version of the package by cloning this GitHub repository and installing the local version of the package on your environment by running `pip install -e .` after navigating to the folder in your terminal.

Packages required to build the documentation are available if you install the software with `pip install micromorph[docs]`. You can then build the documentation by running the command `mkdocs serve`, which will make the documentation available locally (typically at `http://127.0.0.1:8000/`). You can add the flag `--livereload` to ensure the documentation is automatically rebuilt upon editing. 