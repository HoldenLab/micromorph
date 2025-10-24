import time
from qtpy.QtWidgets import (QWidget, QCheckBox, QVBoxLayout,
                            QLineEdit, QLabel, QPushButton, QApplication, QHBoxLayout,
                            QListWidget, QSizePolicy, QTabWidget)
from micromorph.batch.batch_functions import run_omnipose_on_folder, run_analysis_on_folder, run_measure360_on_folder
try:
    from cellpose_omni.models import MODEL_NAMES
except:
    MODEL_NAMES = ["bact_fluor_omni", "bact_phase_omni"]
from ._gui_utils import DropdownMenu, LabelledIntField, LabelledFloatField
from ._MaskFilterWidget import MaskFilterWidget
from superqt import QCollapsible
from micromorph.defaults.defaults import analysis_options, filter_options, export_options, segmentation_options, image_loading_options


class _loadingSettings(QWidget):
    def __init__(self, parent: QWidget):
        super().__init__()
        self.setWindowTitle("Image Loading Settings")
        self._parent = parent

        self.image_reader = DropdownMenu("Image Reader", ["PIL", "tifffile"])
        self.split_axis = DropdownMenu("Split Axis", ["None", "0", "1"])
        self.split_channel = DropdownMenu("Split Channel", ["1", "2"])

        layout = QVBoxLayout()
        layout.addWidget(self.image_reader)
        layout.addWidget(self.split_axis)
        layout.addWidget(self.split_channel)
        self.setLayout(layout)

    def closeEvent(self, event):
        updated_options = {'image_reader': self.returnImageReader(),
                           'split_axis': self.returnSplitAxis(),
                           'split_channel': self.returnSplitChannel()}

        self._parent.image_loading_options.update(updated_options)


    def returnImageReader(self):
        return self.image_reader.returnSelected()


    def returnSplitAxis(self):
        selected_axis = self.split_axis.returnSelected()
        if selected_axis == "None":
            return None
        else:
            return int(selected_axis)


    def returnSplitChannel(self):
        return int(self.split_channel.returnSelected())


class _segmentationSettings(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.setWindowTitle("Segmentation Settings")

        self._parent = parent

        self.params_widget = MaskFilterWidget(None)
        collapsible = QCollapsible("Mask filter options")
        collapsible.collapse(True)
        collapsible.addWidget(self.params_widget)
        collapsible.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)

        # dropdown menu
        self.model_name_dropdown = DropdownMenu("Model Name", MODEL_NAMES)
        idx_selected = MODEL_NAMES.index(self._parent.segmentation_options["model"])
        self.model_name_dropdown.cb.setCurrentIndex(idx_selected)
        self.model_name_dropdown.setContentsMargins(0, 0, 0, 0)

        self.gpu_checkbox = QCheckBox("Use GPU")
        self.gpu_checkbox.setChecked(self._parent.segmentation_options["use_gpu"])


        layout = QVBoxLayout()
        layout.addWidget(self.model_name_dropdown)
        layout.addWidget(self.gpu_checkbox)
        layout.addWidget(collapsible)

        self.setLayout(layout)

        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)

    def closeEvent(self, event):
        updated_parameters = {'model': self.model_name_dropdown.returnSelected(),
                              'use_gpu': self.gpu_checkbox.isChecked(),
                              'binary_filter_options': self.params_widget.get_filter_options()}

        # update dictionary with settings
        self._parent.segmentation_options.update(updated_parameters)


class _filterSettings(QWidget):
    def __init__(self, parent, method):
        super().__init__()
        self._parent = parent
        self.method = method

        self.widgets = []

        self.layout = QVBoxLayout()
        self.layout.setSpacing(0)

        if method == "stdev":
            self.std_dev = LabelledIntField("Std Dev distance from Median", 1)
            self.layout.addWidget(self.std_dev)
            self.widgets.append(self.std_dev)
        elif method == "derivative":
            self.derivative = LabelledIntField("Derivative Threshold", 250)
            self.layout.addWidget(self.derivative)
            self.widgets.append(self.derivative)
        elif method == "sav-gol":
            self.window_size = LabelledIntField("Savitzky-Golay Window Size", 25)
            self.polyorder = LabelledIntField("Savitzky-Golay Polyorder", 3)

            self.layout.addWidget(self.window_size)
            self.layout.addWidget(self.polyorder)

            self.widgets.append(self.window_size)
            self.widgets.append(self.polyorder)

        self.layout.setContentsMargins(0, 0, 0, 0)

        self.setLayout(self.layout)

    def getValues(self):
        values = []
        for widget in self.widgets:
            values.append(widget.getValue())
        return values

    def removeWidget(self):
        print("Removing widgets")
        for widget in self.widgets:
            self.layout.removeWidget(widget)


class _analysisSettings(QWidget):
    def __init__(self, parent, batch_tools=True):
        super().__init__()
        self._parent = parent
        try:
            self._parent.analysis_options
        except AttributeError:
            # Populate with defaults
            self._parents.analysis_options = analysis_options()

        self.setWindowTitle("Analysis Settings")

        self.layout = QVBoxLayout()

        self.pxsize = LabelledFloatField("Pixel Size (nm/px)", self._parent.analysis_options['pxsize'])
        self.psfFWHM = LabelledFloatField("PSF FWHM (nm)", self._parent.analysis_options['psfFWHM'])
        self.n_lines = LabelledIntField("Number of lines", self._parent.analysis_options['n_lines'])

        self.filter_results = QCheckBox("Filter Results")
        self.filter_results.setChecked(self._parent.filter_options["apply_filter"])

        self.fit_type_options = ["fluorescence", "phase", "tophat"]
        self.fit_type_dropdown = DropdownMenu("Fit Type", self.fit_type_options)

        options = ["derivative", "stdev", "None"]

        self.filter_options = DropdownMenu("Filter Options", options)

        if batch_tools:
            # Export options, default both ticked
            self.export_csv = QCheckBox("Export CSV")
            self.export_pickle = QCheckBox("Export Pickle")
            self.export_hdf5 = QCheckBox("Export HDF5")
            self.export_csv.setChecked(True)
            self.export_pickle.setChecked(True)
            self.export_hdf5.setChecked(True)

        idx_selected = options.index(self._parent.filter_options['filter_type'])
        self.filter_options.cb.setCurrentIndex(idx_selected)
        self.filter_parameters = _filterSettings(self, self.filter_options.returnSelected())
        self.filter_options.cb.currentIndexChanged.connect(self.selectionchange)

        fit_type_selected_idx = self.fit_type_options.index(self._parent.analysis_options['fit_type'])
        self.fit_type_dropdown.cb.setCurrentIndex(fit_type_selected_idx)

        if batch_tools:
            export_label = QLabel("Export Options")
            export_label.setStyleSheet("font-weight: bold")
            self.layout.addWidget(export_label)
            self.layout.addWidget(self.export_csv)
            self.layout.addWidget(self.export_pickle)
            self.layout.addWidget(self.export_hdf5)

        self.tabs = QTabWidget()

        self.measure360_tab = QWidget()
        self.measure360_layout = QVBoxLayout()
        self.measure360_tab.setLayout(self.measure360_layout)
        self.measure360_layout.addWidget(self.n_lines)
        self.measure360_layout.addWidget(self.filter_results)
        self.measure360_layout.addWidget(self.filter_options)
        self.measure360_layout.addWidget(self.filter_parameters)
        self.measure360_layout.addStretch()


        self.full_analysis_tab = QWidget()
        self.full_analysis_layout = QVBoxLayout()
        self.full_analysis_tab.setLayout(self.full_analysis_layout)
        self.initialiseFullAnalysisTab()

        self.tabs.addTab(self.full_analysis_tab, "Full Analysis")
        self.tabs.addTab(self.measure360_tab, "Measure360")

        label_analysis = QLabel("Analysis Settings")
        label_analysis.setStyleSheet("font-weight: bold")

        self.layout.addWidget(label_analysis)
        self.layout.addWidget(self.pxsize)
        self.layout.addWidget(self.psfFWHM)
        self.layout.addWidget(self.fit_type_dropdown)
        self.layout.addWidget(self.tabs)

        self.setLayout(self.layout)

    def initialiseFullAnalysisTab(self):
        # Set parameters from the analysis_options dictionary
        self.n_widths = LabelledIntField("Number of Widths", self._parent.analysis_options['n_widths'])
        self.boundary_smoothing_factor = LabelledIntField("Boundary Smoothing Factor", self._parent.analysis_options['boundary_smoothing_factor'])
        self.error_threshold = LabelledFloatField("Error Threshold", self._parent.analysis_options['error_threshold'])
        self.max_iter = LabelledIntField("Max Iterations", self._parent.analysis_options['max_iter'])
        self.min_distance_to_boundary = LabelledIntField("Min Distance to Boundary", self._parent.analysis_options['min_distance_to_boundary'])
        self.step_size = LabelledFloatField("Step Size", self._parent.analysis_options['step_size'])
        self.spline_spacing = LabelledFloatField("Spline Spacing (px)", self._parent.analysis_options['spline_spacing'])
        self.spline_order = LabelledIntField("Spline Order", self._parent.analysis_options['spline_val'])


        self.show_boundaries_checkbox = QCheckBox("Show Boundaries")
        self.show_midlines_checkbox = QCheckBox("Show Midlines")

        self.full_analysis_layout.addWidget(self.n_widths)
        self.full_analysis_layout.addWidget(self.boundary_smoothing_factor)
        self.full_analysis_layout.addWidget(self.error_threshold)
        self.full_analysis_layout.addWidget(self.max_iter)
        self.full_analysis_layout.addWidget(self.min_distance_to_boundary)
        self.full_analysis_layout.addWidget(self.step_size)
        self.full_analysis_layout.addWidget(self.spline_spacing)
        self.full_analysis_layout.addWidget(self.spline_order)
        self.full_analysis_layout.addWidget(self.show_boundaries_checkbox)
        self.full_analysis_layout.addWidget(self.show_midlines_checkbox)

    def selectionchange(self, i):
        print(self.filter_options.returnSelected())
        self.layout.removeWidget(self.filter_parameters)
        self.filter_parameters = _filterSettings(self, self.filter_options.returnSelected())
        self.layout.addWidget(self.filter_parameters)

    # Print "close" when the window is closed
    def closeEvent(self, event):
        # Update parent.options

        updated_parameters = {'pxsize': self.pxsize.getValue(),
                              'psfFWHM': self.psfFWHM.getValue(),
                              'n_lines': self.n_lines.getValue(),
                              'n_widths': self.n_widths.getValue(),
                              'boundary_smoothing_factor': self.boundary_smoothing_factor.getValue(),
                              'error_threshold': self.error_threshold.getValue(),
                              'max_iter': self.max_iter.getValue(),
                              'min_distance_to_boundary': self.min_distance_to_boundary.getValue(),
                              'step_size': self.step_size.getValue(),
                              'fit_type': self.fit_type_dropdown.returnSelected(),
                              'show_boundaries': self.show_boundaries_checkbox.isChecked(),
                              'show_midlines': self.show_midlines_checkbox.isChecked(),
                              'spline_val': self.spline_order.getValue(),
                              'spline_spacing': self.spline_spacing.getValue()}

        self._parent.analysis_options.update(updated_parameters)

        updated_filter_options = {'apply_filter': self.filter_results.isChecked(),
                                  'filter_type': self.filter_options.returnSelected(),
                                  'filter_settings': self.filter_parameters.getValues()}

        self._parent.filter_options.update(updated_filter_options)

        if hasattr(self, 'export_csv'):
            self._parent.export_options.update({'export_type': self.check_output_type()})

    def check_output_type(self):
        # Check which checkboxes are ticked, if both are ticked, return 'both'
        if self.export_csv.isChecked() and self.export_pickle.isChecked() and self.export_hdf5.isChecked():
            return 'all'
        elif self.export_csv.isChecked():
            return 'csv'
        elif self.export_pickle.isChecked():
            return 'pickle'
        elif self.export_hdf5.isChecked():
            return 'hdf5'
        else:
            return 'none'


class _folderString(QWidget):
    def __init__(self, label, text):
        super().__init__()
        layout = QHBoxLayout()
        self.label = QLabel(label)
        self.edit_line = QLineEdit(text)
        layout.addWidget(self.label)
        layout.addWidget(self.edit_line)
        self.setLayout(layout)


class _checkboxPair(QWidget):
    def __init__(self, options, default):
        super().__init__()
        layout = QVBoxLayout()
        self.checkboxes = [QCheckBox(option) for option in options]
        for checkbox in self.checkboxes:
            layout.addWidget(checkbox)

        # Set first checkbox to checked
        self.checkboxes[default].setChecked(True)
        self.setLayout(layout)

        self.checkboxes[0].stateChanged.connect(lambda: self.set_status_to_opposite(0))
        self.checkboxes[1].stateChanged.connect(lambda: self.set_status_to_opposite(1))

    def set_status_to_opposite(self, a):
        # Get the status of the checkbox which just changed, set the other one to the opposite
        if self.checkboxes[a].isChecked():
            self.checkboxes[1-a].setChecked(False)
        else:
            self.checkboxes[1-a].setChecked(True)


class _batchtoolsPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        self.input_directory = _folderString("Input Directory", "")
        self.output_directory = _folderString("Output Directory", "")

        # Create checkboxes
        segmentation_options_labels = ["Run Segmentation", "Load Segmentation"]
        self.segmentation_checkboxes = _checkboxPair(segmentation_options_labels, 0)

        analysis_options_labels = ["Full Analysis", "Measure360"]
        self.analysis_checkboxes = _checkboxPair(analysis_options_labels, 1)

        self.pil_image_reading_checkbox = QCheckBox("Use pil image reader.")

        buttons_layout = QHBoxLayout()
        self.segmentation_settings_button = QPushButton("Segmentation Settings")
        self.analysis_settings_button = QPushButton("Analysis Settings")
        buttons_layout.addWidget(self.segmentation_settings_button)
        buttons_layout.addWidget(self.analysis_settings_button)

        run_buttons_layout = QHBoxLayout()
        self.run_segmentation_button = QPushButton("Run Segmentation")
        self.run_analysis_button = QPushButton("Run Analysis")
        run_buttons_layout.addWidget(self.run_segmentation_button)
        run_buttons_layout.addWidget(self.run_analysis_button)

        self.run_button = QPushButton("Run")
        self.loading_settings_button = QPushButton("Image Loading Settings")

        self.add_button = QPushButton("Add Folder")
        self.add_button.clicked.connect(self.add_to_list)

        self.remove_button = QPushButton("Remove Folder")
        self.remove_button.clicked.connect(self.remove_from_list)

        self.listwidg = QListWidget()

        # ---- Add widgets to layout ----
        layout.addWidget(self.input_directory)
        layout.addWidget(self.add_button)
        layout.addWidget(self.remove_button)
        layout.addWidget(self.listwidg)
        layout.addWidget(self.segmentation_checkboxes)
        layout.addWidget(self.analysis_checkboxes)
        layout.addWidget(self.loading_settings_button)
        layout.addLayout(buttons_layout)
        layout.addLayout(run_buttons_layout)
        layout.addWidget(self.run_button)

        self.setLayout(layout)

        # ---- Set default options ----
        self.analysis_options = analysis_options()
        self.filter_options = filter_options()
        self.export_options = export_options()
        self.segmentation_options = segmentation_options()
        self.image_loading_options = image_loading_options()

        # ---- Connect operations ----
        self.analysis_settings_button.clicked.connect(self.open_analysis_settings)
        self.segmentation_settings_button.clicked.connect(self.open_segmentation_settings)
        self.run_analysis_button.clicked.connect(self.run_analysis)
        self.run_segmentation_button.clicked.connect(self.run_segmentation)
        self.run_button.clicked.connect(self.run_all)
        self.loading_settings_button.clicked.connect(self.open_loading_settings)

    def open_loading_settings(self):
        self.loading_settings = _loadingSettings(self)
        self.loading_settings.setStyleSheet(self.styleSheet())
        self.loading_settings.show()

    def open_analysis_settings(self):
        self.analysis_settings = _analysisSettings(self)
        self.analysis_settings.setStyleSheet(self.styleSheet())
        self.analysis_settings.show()

    def open_segmentation_settings(self):
        self.segmentation_settings = _segmentationSettings(self)
        self.segmentation_settings.setStyleSheet(self.styleSheet())
        self.segmentation_settings.show()

    def run_segmentation(self):
        folder_list = self.get_folder_list()

        for i, folder in enumerate(folder_list):
            # folder_name, model_name, gpu_option = False, chans = None, verbose = False
            print("Running segmentation on folder: " + folder)
            run_omnipose_on_folder(folder,
                                   self.segmentation_options["model"],
                                   self.segmentation_options["use_gpu"],
                                   chans=None,
                                   verbose=False,
                                   mask_filter_options=self.segmentation_options["binary_filter_options"],
                                   loading_method=self.image_loading_options["image_reader"],
                                   split_axis=self.image_loading_options["split_axis"],
                                   split_channel=self.image_loading_options["split_channel"])

    def run_analysis(self):
        folder_list = self.get_folder_list()

        for i, folder in enumerate(folder_list):
            # folder_name, options = dict()
            options = {'save': True,
                       'save_directory': folder,
                       'measure360': self.analysis_checkboxes.checkboxes[1].isChecked()}
            self.analysis_options.update(options)

            print("Running analysis on folder: " + folder)
            start_time = time.time()

            if self.analysis_options['measure360']:
                run_measure360_on_folder(folder,
                                         self.analysis_options,
                                         self.export_options,
                                         self.image_loading_options,
                                         self.filter_options)
            else:
                run_analysis_on_folder(folder,
                                       self.analysis_options)

            end_time = time.time()
            print("Analysis complete. Time elapsed: " + str(end_time - start_time) + " seconds")

    def run_all(self):
        if self.segmentation_checkboxes.checkboxes[0].isChecked():
            self.run_segmentation()
        self.run_analysis()

    def add_to_list(self):
        text = self.input_directory.edit_line.text()
        self.listwidg.addItem(text)

    def remove_from_list(self):
        self.listwidg.takeItem(self.listwidg.currentRow())

    def get_folder_list(self):
        # get all items in listwidg
        items = []
        for i in range(self.listwidg.count()):
            items.append(self.listwidg.item(i).text())

        return items
    

def open_micromorph_batch_gui():
    # set up app
    app = QApplication([])

    # Set app title
    app.setApplicationName("MicroMorph Batch")

    batch_tools_widget = _batchtoolsPanel()
    batch_tools_widget.show()

    app.exec_()