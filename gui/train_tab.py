import os
import json
import importlib
from pathlib import Path

from PyQt5.QtWidgets import (QWidget, QComboBox, QGridLayout,
                             QVBoxLayout, QLabel, QLineEdit, QPushButton,
                             QApplication)
from PyQt5.QtGui import (QIntValidator)

class TrainTab(QWidget):

    def __init__(self):
        super().__init__()

        self.available_models = []
        self.model = None
        self.model_dir = None
        self.weights_name = None
        self.available_data = []
        self.data_dir = None
        self.batch_size = 50
        self.epochs = 100

        self.initUI()
    
    def initUI(self):
        self.train_button = QPushButton("Train")
        self.train_button.toggle()
        self.train_button.setCheckable(True)
        self.train_button.setEnabled(False)
        self.train_button.clicked.connect(self.toggle_train_button)

        self.weights_name_field = QLineEdit()
        self.weights_name_field.textChanged.connect(self.weights_name_changed)
        self.weights_name_field.setEnabled(False)

        self.data_selection = QComboBox()
        self.data_selection.setEnabled(False)
        self.data_selection.activated.connect(self.select_data)

        batch_size_widget = QLineEdit(str(self.batch_size))
        batch_size_widget.setValidator(QIntValidator())
        batch_size_widget.textChanged.connect(lambda new_val: self.line_edit_value_changed(new_val, "batch_size"))

        epochs_widget = QLineEdit(str(self.epochs))
        epochs_widget.setValidator(QIntValidator())
        epochs_widget.textChanged.connect(lambda new_val: self.line_edit_value_changed(new_val, "epochs"))

        model_selection = self.init_model_selection()

        menu_widget = QWidget()
        menu_layout = QGridLayout()
        menu_layout.setColumnStretch(1, 1)
        menu_layout.setColumnStretch(2, 1)
        menu_layout.setColumnStretch(3, 3)
        menu_layout.addWidget(QWidget())
        menu_layout.addWidget(QLabel("Select model: "), 1, 1)
        menu_layout.addWidget(model_selection, 1, 2)
        menu_layout.addWidget(QLabel("Select data: "), 2, 1)
        menu_layout.addWidget(self.data_selection, 2, 2)
        menu_layout.addWidget(QLabel("Batch size: "), 3, 1)
        menu_layout.addWidget(batch_size_widget, 3, 2)
        menu_layout.addWidget(QLabel("Epochs: "), 4, 1)
        menu_layout.addWidget(epochs_widget, 4, 2)
        menu_layout.addWidget(QLabel("Name for weights: "), 5,1)
        menu_layout.addWidget(self.weights_name_field, 5,2)
        menu_layout.addWidget(self.train_button, 6, 1)
        menu_layout.addWidget(QWidget())
        menu_widget.setLayout(menu_layout)
        
        main_layout = QVBoxLayout()
        main_layout.addWidget(menu_widget)
        self.setLayout(main_layout)

    def init_model_selection(self):        
        model_selection = QComboBox()
        model_selection.activated.connect(self.select_model)
        models_path = Path(os.getcwd()) / "models"
        for model_folder in models_path.iterdir():
            if not model_folder.is_dir() or model_folder.name == "__pycache__":
                continue
            model_selection.addItem(model_folder.name)
            self.available_models.append(model_folder)
        if self.available_models:
            self.select_model(0)     
        return model_selection

    def toggle_train_button(self):
        if self.train_button.isChecked():
            self.train_button.setStyleSheet("background-color: orange")
            self.train_button.setEnabled(False)
            QApplication.processEvents()
            self.train()

    def select_model(self, idx):
        self.model_dir = self.available_models[idx]
        module_path = str((self.model_dir / "model.py").absolute())
        spec = importlib.util.spec_from_file_location("model", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model = module.Model()
        self.weights_name_field.setEnabled(True)
        self.refresh_data_selection()

    def weights_name_changed(self, new_name):
        if not self.weights_name:
            self.train_button.setEnabled(True)
        self.weights_name = new_name

    def refresh_data_selection(self):
        self.data_selection.setEnabled(True)
        self.available_data = []
        data_path = self.model_dir / "data"
        self.data_selection.clear()
        for data in data_path.iterdir():
            if not data.is_dir() or data.name == "__pycache__":
                continue
            self.data_selection.addItem(data.name)
            self.available_data.append(data)
        if self.available_data:
            self.select_data(0)

    def select_data(self, idx):
        self.data_dir = self.available_data[idx] 


    def train(self):
        self.model.load_data(self.data_dir)
        self.model.create_model()
        self.model.train(batch_size=self.batch_size, epochs=self.epochs, weights_name=self.weights_name)
        self.train_button.setChecked(False)
        self.train_button.setEnabled(True)
        self.train_button.setStyleSheet("")    
        
    def line_edit_value_changed(self, new_val, variable_name):
        try:
            # Segmentation fault if screen_capture_size set to 1
            if int(new_val) > 1:
                setattr(self, variable_name, int(new_val))
        except ValueError:
            pass


