from qtpy.QtWidgets import (QPushButton, QWidget, QGridLayout,
                            QVBoxLayout, QCheckBox)
from ._gui_utils import LabelledIntField, _layerSelectionWidget
from micromorph.segmentation import filter_mask


class MaskFilterWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()

        if napari_viewer:
            self.viewer = napari_viewer
            # Create dropdown menu for layer selection
            self.layer_selection_dropdown = _layerSelectionWidget(self.viewer, layer_type=napari.layers.labels.labels.Labels)

        # Checkboxes for filtering
        self.min_area_checkbox = QCheckBox()
        self.max_area_checkbox = QCheckBox()
        self.min_length_checkbox = QCheckBox()
        self.max_length_checkbox = QCheckBox()
        self.min_width_checkbox = QCheckBox()
        self.max_width_checkbox = QCheckBox()

        # Create fields for the user to input parameters
        self.min_area = LabelledIntField("Minimum area", 0)
        self.max_area = LabelledIntField("Maximum area", 10000)

        self.min_length = LabelledIntField("Minimum length", 0)
        self.max_length = LabelledIntField("Maximum length", 1000)

        self.min_width = LabelledIntField("Minimum width", 0)
        self.max_width = LabelledIntField("Maximum width", 1000)

        if napari_viewer:
            # Create buttons
            self.run_button = QPushButton("Apply filter")

        # Layout
        main_layout = QVBoxLayout()
        layout = QGridLayout()
        main_layout.addLayout(layout)

        if napari_viewer:
            layout.addWidget(self.layer_selection_dropdown, 0, 0, 1, 2)

        layout.addWidget(self.min_area, 1, 0, 1, 1)
        layout.addWidget(self.max_area, 2, 0, 1, 1)
        layout.addWidget(self.min_length, 3, 0, 1, 1)
        layout.addWidget(self.max_length, 4, 0, 1, 1)
        layout.addWidget(self.min_width, 5, 0, 1, 1)
        layout.addWidget(self.max_width, 6, 0, 1, 1)

        layout.addWidget(self.min_area_checkbox, 1, 1, 1, 1)
        layout.addWidget(self.max_area_checkbox, 2, 1, 1, 1)
        layout.addWidget(self.min_length_checkbox, 3, 1, 1, 1)
        layout.addWidget(self.max_length_checkbox, 4, 1, 1, 1)
        layout.addWidget(self.min_width_checkbox, 5, 1, 1, 1)
        layout.addWidget(self.max_width_checkbox, 6, 1, 1, 1)

        if napari_viewer:
            layout.addWidget(self.run_button, 7, 0, 1, 2)

        self.setLayout(main_layout)

        if napari_viewer:
            # Connect operations
            self.run_button.clicked.connect(self.run_filter_on_selected_mask)

    def run_filter_on_selected_mask(self):
        # Get selected layer
        layer_name = self.layer_selection_dropdown.layer_selection.returnSelected()

        label_data = self.viewer.layers[layer_name].data

        filter_options = self.get_filter_options()

        filtered_label_data = filter_mask(label_data, options=filter_options)

        # create a new labels layer
        results_name = "Filtered labels of " + layer_name
        self.viewer.add_labels(filtered_label_data, name=results_name)

    def get_filter_options(self):
        if self.min_area_checkbox.isChecked():
            min_area = self.min_area.getValue()
        else:
            min_area = None

        if self.max_area_checkbox.isChecked():
            max_area = self.max_area.getValue()
        else:
            max_area = None

        if self.min_length_checkbox.isChecked():
            min_length = self.min_length.getValue()
        else:
            min_length = None

        if self.max_length_checkbox.isChecked():
            max_length = self.max_length.getValue()
        else:
            max_length = None

        if self.min_width_checkbox.isChecked():
            min_width = self.min_width.getValue()
        else:
            min_width = None

        if self.max_width_checkbox.isChecked():
            max_width = self.max_width.getValue()
        else:
            max_width = None

        filter_options = {'min_area': min_area,
                          'max_area': max_area,
                          'min_length': min_length,
                          'max_length': max_length,
                          'min_width': min_width,
                          'max_width': max_width}

        return filter_options
    