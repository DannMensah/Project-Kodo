import os
import json
import importlib
from pathlib import Path

from PyQt5.QtWidgets import (QWidget, QSplitter, QLabel, QGridLayout,
                             QVBoxLayout, QComboBox, QPushButton, QLineEdit,
                             QCheckBox, QApplication)
from PyQt5.QtGui import (QPixmap, QImage) 
from PyQt5.QtCore import (Qt)

class ProcessTab(QWidget):

    def __init__(self):
        super().__init__()

        self.available_data_folders = []
        self.data_folder = None
        self.available_models = []
        self.model = None
        self.input_channels_mask = []
        self.info = None
        self.show_data_screen = True

        self.initUI()
    
    def initUI(self):
        self.screen_label = QLabel()

        channels_widget = QWidget()
        self.channels_layout = QGridLayout()
        channels_widget.setLayout(self.channels_layout)

        show_data_screen_widget = QCheckBox("Show processed images (might slow down processing)")
        show_data_screen_widget.setChecked(True)
        show_data_screen_widget.stateChanged.connect(self.data_screen_toggled)
        
        screen_layout = QVBoxLayout()
        screen_layout_widget = QWidget()
        screen_layout.addWidget(self.screen_label)
        screen_layout.addWidget(show_data_screen_widget)
        screen_layout_widget.setLayout(screen_layout)


        channels_screen_splitter = QSplitter(Qt.Horizontal)
        channels_screen_splitter.addWidget(screen_layout_widget)
        channels_screen_splitter.addWidget(channels_widget)
        channels_screen_splitter.setSizes([400,200])
        channels_screen_splitter.setCollapsible(1, False)

        self.process_button = QPushButton("Process")
        self.process_button.toggle()
        self.process_button.setCheckable(True)
        self.process_button.clicked.connect(self.toggle_process_button)
        self.process_button.setEnabled(False)
        data_folder_selection = self.init_data_folder_selection()
        model_selection = self.init_model_selection()
        
        menu_widget = QWidget()
        menu_layout = QGridLayout()
        menu_layout.setColumnStretch(1, 1)
        menu_layout.setColumnStretch(2, 1)
        menu_layout.setColumnStretch(3, 3)
        menu_layout.addWidget(QLabel("Select raw data folder"), 1, 1, Qt.AlignRight)
        menu_layout.addWidget(data_folder_selection, 1,2, Qt.AlignRight)
        menu_layout.addWidget(QLabel("Select target model"), 2, 1, Qt.AlignRight)
        menu_layout.addWidget(model_selection, 2,2, Qt.AlignRight)
        menu_layout.addWidget(self.process_button, 3,1)
        menu_widget.setLayout(menu_layout)

        main_layout = QVBoxLayout()
        main_layout.addWidget(channels_screen_splitter)
        main_layout.addWidget(menu_widget)
        self.setLayout(main_layout)

    def init_data_folder_selection(self):
        data_folder_selection = QComboBox()
        data_folder_selection.activated.connect(self.select_data_folder)
        data_folder = Path(os.getcwd()) / "data"
        data_folder_selection.addItem("Process all non-processed")
        for data in data_folder.iterdir():
            if not data.is_dir() or data.name == "__pycache__":
                continue
            data_folder_selection.addItem(data.name)
            self.available_data_folders.append(data)
        if len(self.available_data_folders) > 1:
            self.select_data_folder(0)
        else:
            data_folder_selection.setEnabled(False)
        return data_folder_selection

    def data_screen_toggled(self):
        self.show_data_screen = not self.show_data_screen

    def select_data_folder(self, idx):
        if idx == 0:
            self.data_folder = self.available_data_folders[1]
            self.info, key_labels_unchanged = self.load_info()
            self.refresh_key_mask(key_labels_unchanged)
            self.data_folder = None
        else:
            self.data_folder = self.available_data_folders[idx]
            self.info, key_labels_unchanged = self.load_info()
            self.refresh_key_mask(key_labels_unchanged)
        if self.model:
            self.process_button.setEnabled(True)

    def load_info(self):
        with open(self.data_folder / "info.json") as info_file:
            new_info = json.load(info_file)
        key_labels_unchanged = self.info and new_info["key_labels"] == self.info["key_labels"]
        return new_info, key_labels_unchanged

    def clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget() is not None:
                child.widget().deleteLater()
            elif child.layout() is not None:
                clearLayout(child.layout())

    def refresh_key_mask(self, key_labels_unchanged):
        if key_labels_unchanged:
            return
        key_labels = self.info["key_labels"]
        self.input_channels_mask = [False] * len(key_labels)
        self.clear_layout(self.channels_layout)
        for idx, label in enumerate(key_labels):
            checkbox_widget = QCheckBox(label)
            checkbox_widget.setChecked(False)
            checkbox_widget.stateChanged.connect(lambda checked, idx=idx: self.checkbox_checked(checked, idx))
            self.channels_layout.addWidget(checkbox_widget)

    def checkbox_checked(self, checked, idx):
        self.input_channels_mask[idx] = not self.input_channels_mask[idx]

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

    def select_model(self, idx):
        self.model_dir = self.available_models[idx]
        module_path = str((self.model_dir / "model.py").absolute())
        spec = importlib.util.spec_from_file_location("model", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model = module.KodoModel()
        if self.data_folder:
            self.process_button.setEnabled(True)

    def toggle_process_button(self):
        if self.process_button.isChecked():
            self.process_button.setStyleSheet("background-color: blue")
            self.process_button.setEnabled(False)
            self.process()

    def update_image(self, img):
        if self.show_data_screen:
            height, width, channel = img.shape
            bytes_per_line = 3 * width
            q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
            q_pixmap = QPixmap.fromImage(q_img).scaled(self.screen_label.width(), 
                                                       self.screen_label.height(), 
                                                       Qt.KeepAspectRatio)
            self.screen_label.setPixmap(q_pixmap)
            self.screen_label.show()
        QApplication.processEvents()

    def process(self):
        if not self.data_folder:
            for idx, folder in enumerate(self.available_data_folders):
                if idx != 0:
                    self.data_folder = self.available_data_folders[idx]
                    self.info, key_labels_unchanged = self.load_info()
                    if not key_labels_unchanged:
                        continue
                    self.model.process(self.data_folder, 
                                       self.input_channels_mask, 
                                       img_update_callback=self.update_image)
        else:
            self.model.process(self.data_folder, self.input_channels_mask, img_update_callback=self.update_image)
        self.process_button.setChecked(False)
        self.process_button.setEnabled(True)
        self.process_button.setStyleSheet("")

    



