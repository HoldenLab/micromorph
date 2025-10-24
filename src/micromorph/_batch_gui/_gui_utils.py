from qtpy.QtGui import QIntValidator
from qtpy.QtWidgets import QWidget, QLabel, QLineEdit, QFormLayout, QHBoxLayout, QComboBox

class LabelledIntField(QWidget):
    def __init__(self, title, initial_value=None):
        QWidget.__init__(self)

        self.label = QLabel()
        self.label.setText(title)
        self.label.setContentsMargins(0, 0, 0, 0)
        # self.label.setFixedWidth(100)
        # layout.addWidget(self.label)

        self.lineEdit = QLineEdit(self)
        self.lineEdit.setContentsMargins(0, 0, 0, 0)
        # self.lineEdit.setFixedWidth(40)
        self.lineEdit.setValidator(QIntValidator())
        if initial_value != None:
            self.lineEdit.setText(str(initial_value))

        layout = QFormLayout()
        self.setLayout(layout)
        layout.addRow(QLabel(title), self.lineEdit)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setContentsMargins(0, 0, 0, 0)

    def setLabelWidth(self, width):
        self.label.setFixedWidth(width)

    def setInputWidth(self, width):
        self.lineEdit.setFixedWidth(width)

    def getValue(self):
        return int(self.lineEdit.text())

class LabelledFloatField(QWidget):
    def __init__(self, title, initial_value=None):
        QWidget.__init__(self)

        self.label = QLabel()
        self.label.setText(title)
        self.label.setContentsMargins(0, 0, 0, 0)
        # self.label.setFixedWidth(100)
        # layout.addWidget(self.label)

        self.lineEdit = QLineEdit(self)
        self.lineEdit.setContentsMargins(0, 0, 0, 0)
        # self.lineEdit.setFixedWidth(40)
        self.lineEdit.setValidator(QIntValidator())
        if initial_value != None:
            self.lineEdit.setText(str(initial_value))

        layout = QFormLayout()
        self.setLayout(layout)
        layout.addRow(QLabel(title), self.lineEdit)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setContentsMargins(0, 0, 0, 0)

    def setLabelWidth(self, width):
        self.label.setFixedWidth(width)

    def setInputWidth(self, width):
        self.lineEdit.setFixedWidth(width)

    def getValue(self):
        return float(self.lineEdit.text())

class DropdownMenu(QWidget):
    def __init__(self, label, options):
        QWidget.__init__(self)
        layout = QHBoxLayout()

        self.label = QLabel()
        self.label.setText(label)

        self.cb = QComboBox()
        self.cb.addItems(options)

        layout.addWidget(self.label)
        layout.addWidget(self.cb)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)


    def updateMenu(self, options):
        self.cb.clear()
        self.cb.addItems(options)


    def returnSelected(self):
        return self.cb.currentText()


class _layerSelectionWidget(QWidget):
    def __init__(self, napari_viewer, label="Select layer", layer_type=None):
        super().__init__()
        self.viewer = napari_viewer
        self._type = layer_type

        self.getLayerList()
        # self.layer_list = self.viewer.layers
        #
        # if layer_type is not None:
        #     self.layer_list = [layer for layer in self.layer_list if isinstance(layer, self._type)]

        self.layer_selection = DropdownMenu(label, [layer.name for layer in self.layer_list])

        layout = QHBoxLayout()

        layout.addWidget(self.layer_selection)

        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.viewer.layers.events.connect(self.updateLayerList)

    def getLayerList(self):
        all_layers_list = self.viewer.layers
        self.layer_list = [layer for layer in all_layers_list if isinstance(layer, self._type)]

    def returnSelectedLayer(self):
        return self.layer_selection.returnSelected()

    def updateLayerList(self):
        self.getLayerList()
        self.layer_selection.updateMenu([layer.name for layer in self.layer_list])
