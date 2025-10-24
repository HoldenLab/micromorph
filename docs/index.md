# Welcome to MicroMorph

`micromorph` is a python package designed to perform morphological measurements of bacterial microscopy images acquired using either phase contrast, or fluorescence microscopy. 

`micromorph` can be used as a package, or through the napari plugin `napari-micromorph`.

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

## Usage
### Napari plugin
<!-- The plugin can be accessed through the napari GUI. Once [installed](https://napari.org/dev/tutorials/fundamentals/installation.html), you can access the plugin by clicking on the `Plugins` menu, and selecting `Bacteria Morphology Toolkit`. -->
### Standalone batch processing
<!-- The batch tools panel can be run independently of napari, using the following snippet:

```python
from napari_bacteria_morphology_toolkit import _batchtoolsPanel
from qtpy.QtWidgets import QApplication

# set up app
app = QApplication([])

batch_tools_widget = _batchtoolsPanel()
batch_tools_widget.show()

app.exec_()
```

or from the command line, using 
    
```bash
napari-bactmeasure
```

for the napari gui with the toolkit and

```bash
batchtool
```

for the standalone batch processing tool. -->
### API
<!-- The API can be used to perform measurements on images without the need for napari. See the API documentation for 
more information, and the examples for how to use the API. -->