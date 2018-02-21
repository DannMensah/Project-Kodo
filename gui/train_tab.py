import os
import json
import importlib
from pathlib import Path

import numpy as np
from PyQt5.QtWidgets import (QWidget, QComboBox, QGridLayout,
                             QVBoxLayout, QLabel, QLineEdit, QPushButton,
                             QApplication, QListWidget)
from PyQt5.QtGui import (QIntValidator)
from PyQt5.QtCore import (Qt)

class TrainTab(QWidget):

    def __init__(self):
        super().__init__()

        self.available_models = []
        self.model = None
        self.model_dir = None
        self.weights_name = None
        self.highlighted_data = None
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

        self.data_list = QListWidget()
        self.data_list.setEnabled(False)
        self.data_list.itemClicked.connect(self.highlight_available_data)

        self.added_data_list = QListWidget()
        self.added_data_list.setEnabled(False)
        self.added_data_list.itemClicked.connect(self.highlight_added_data_list)

        self.move_data_button = QPushButton("<< >>")
        self.move_data_button.clicked.connect(self.move_data)
        self.move_data_button.setEnabled(False)

        batch_size_widget = QLineEdit(str(self.batch_size))
        batch_size_widget.setValidator(QIntValidator())
        batch_size_widget.textChanged.connect(lambda new_val: self.line_edit_value_changed(new_val, "batch_size"))

        epochs_widget = QLineEdit(str(self.epochs))
        epochs_widget.setValidator(QIntValidator())
        epochs_widget.textChanged.connect(lambda new_val: self.line_edit_value_changed(new_val, "epochs"))

        model_list = self.init_model_list()

        menu_widget = QWidget()
        menu_layout = QGridLayout()
        menu_layout.setColumnStretch(1, 1)
        menu_layout.setColumnStretch(2, 2)
        menu_layout.setColumnStretch(3, 1)
        menu_layout.setColumnStretch(4, 2)
        menu_layout.setColumnStretch(5, 2)
        menu_layout.addWidget(QWidget())
        menu_layout.addWidget(QLabel("Select model: "), 1, 1)
        menu_layout.addWidget(model_list, 1, 2)
        menu_layout.addWidget(QLabel("Available data:"), 2, 2, Qt.AlignCenter)
        menu_layout.addWidget(QLabel("Selected data:"), 2, 4, Qt.AlignCenter)
        menu_layout.addWidget(QLabel("Select data: "), 3, 1)
        menu_layout.addWidget(self.data_list, 3, 2)
        menu_layout.addWidget(self.move_data_button, 3, 3)
        menu_layout.addWidget(self.added_data_list, 3, 4)        
        menu_layout.addWidget(QLabel("Batch size: "), 4, 1)
        menu_layout.addWidget(batch_size_widget, 4, 2)
        menu_layout.addWidget(QLabel("Epochs: "), 5, 1)
        menu_layout.addWidget(epochs_widget, 5, 2)
        menu_layout.addWidget(QLabel("Name for weights: "), 6,1)
        menu_layout.addWidget(self.weights_name_field, 6,2)
        menu_layout.addWidget(self.train_button, 7, 1)
        menu_layout.addWidget(QWidget())
        menu_widget.setLayout(menu_layout)
        
        main_layout = QVBoxLayout()
        main_layout.addWidget(menu_widget)
        self.setLayout(main_layout)

    def init_model_list(self):        
        model_list = QComboBox()
        model_list.activated.connect(self.select_model)
        models_path = Path(os.getcwd()) / "models"
        for model_folder in models_path.iterdir():
            if not model_folder.is_dir() or model_folder.name == "__pycache__":
                continue
            model_list.addItem(model_folder.name)
            self.available_models.append(model_folder)
        if self.available_models:
            self.select_model(0)     
        return model_list

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
        self.refresh_data_list()

    def weights_name_changed(self, new_name):
        if not self.weights_name:
            self.train_button.setEnabled(True)
        self.weights_name = new_name

    def refresh_data_list(self):
        self.move_data_button.setEnabled(True)
        self.added_data_list.setEnabled(True)
        self.data_list.setEnabled(True)
        data_path = self.model_dir / "data"
        self.data_list.clear()
        for data in data_path.iterdir():
            if not data.is_dir() or data.name == "__pycache__":
                continue
            self.data_list.addItem(data.name)

    def highlight_available_data(self, item):
        self.highlighted_data = item.text()
        self.added_data_list.clearSelection()
    
    def highlight_added_data_list(self, item):
        self.highlighted_data = item.text()
        self.data_list.clearSelection()

    def move_data(self):
        if self.is_in_qlist(self.highlighted_data, self.data_list):
            self.remove_from_qlist(self.highlighted_data, self.data_list)
            self.added_data_list.addItem(self.highlighted_data)
            if self.data_list.currentItem():
                self.highlighted_data = self.data_list.currentItem().text()
        else:
            self.remove_from_qlist(self.highlighted_data, self.added_data_list)
            self.data_list.addItem(self.highlighted_data)
            if self.added_data_list.currentItem():
                self.highlighted_data = self.added_data_list.currentItem().text()


    def is_in_qlist(self, text, qlist):
        for idx in range(qlist.count()):
            item = qlist.item(idx)
            if item.text() == text:
                return True
        return False

    def remove_from_qlist(self, text, qlist):
        for idx in range(qlist.count()):
            item = qlist.item(idx)
            if item.text() == text:
                qlist.takeItem(idx)
                break

    def get_added_data_names(self):
        names = []
        for idx in range(self.added_data_list.count()):
            name = self.added_data_list.item(idx).text()
            names.append(name)
        return names

    def stack_data(self):
        X_list = []
        y_list = []
        info = None

        for name in self.get_added_data_names():
            data_path = self.model_dir / "data" / name
            X_list.append(np.load(data_path / "X.npy"))
            y_list.append(np.load(data_path / "y.npy"))
        with open(data_path / "info.json") as fp:
            info = json.load(fp)
        return (np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0), info)

    def train(self):
        self.model.X, self.model.y, self.model.info = self.stack_data()
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


