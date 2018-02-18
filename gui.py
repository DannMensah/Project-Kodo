import sys
import os
import json
import importlib
from pathlib import Path

import numpy as np
import time
import pygame
from skimage.transform import resize
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout,
                             QHBoxLayout, QWidget, QComboBox, QPushButton, QSplitter, QFrame,
                             QTextEdit, QTabWidget, QFileDialog, QGridLayout, QLineEdit)
from PyQt5.QtGui import (QPixmap, QImage)
from PyQt5.QtCore import (Qt, QTimer)

import recorder
from utilities import try_make_dirs
from controller_mappings import PYGAME_TO_XBOX
from controllers import PyvJoyXboxController

class Window(QTabWidget):

    def __init__(self):
        super().__init__()

        self.recording = False
        self.predicting = False
        self.frame = 1
        self.available_models = []
        self.model_dir = None
        self.available_weights = None 
        self.weights = None
        self.controller = None

        self.tab_record = QWidget()
        self.tab_process = QWidget()

        self.addTab(self.tab_record, "Record/Predict")
        self.addTab(self.tab_process, "Process")
        
        self.init_record_UI()
        self.init_record_loop()

        self.setGeometry(1000, 0, 1000, 800)
        self.setMinimumWidth(300)
        self.setMinimumHeight(500)
        self.setWindowTitle('Project Kodo')
        self.updater.start()
        self.show()

    def init_record_UI(self):
        self.save_dir = None
        self.input_source = None

        self.record_button = QPushButton("Record")
        self.predict_button = QPushButton("Predict")

        recorder.init_gamepad_capture()

        output_keys_widget = QWidget()
        self.output_keys_layout = QGridLayout()
        output_keys_widget.setLayout(self.output_keys_layout)

        menu_widget = QWidget()
        self.record_screen_label = QLabel(self)
        self.weights_selection = QComboBox()
        self.weights_selection.setEnabled(False)
        self.weights_selection.activated.connect(self.select_weights)
        input_selection = self.init_input_selection()
        model_selection = self.init_model_selection()
        self.file_path_widget = QLineEdit(os.getcwd())
        self.file_path_widget.setEnabled(False)
        save_button = QPushButton("Choose...")
        save_button.clicked.connect(self.get_save_dir)
        self.record_button.toggle()
        self.record_button.setCheckable(True)
        self.record_button.clicked.connect(self.toggle_record_button)
        self.predict_button.toggle()
        self.predict_button.setCheckable(True)
        self.predict_button.clicked.connect(self.toggle_predict_button)

        menu_layout = QGridLayout()
        menu_layout.setColumnStretch(1, 1)
        menu_layout.setColumnStretch(2, 3)
        menu_layout.setColumnStretch(3, 1)
        menu_layout.addWidget(QWidget())
        menu_layout.addWidget(QLabel("Input device:"), 1, 1, Qt.AlignRight)
        menu_layout.addWidget(input_selection, 1, 2)
        menu_layout.addWidget(QLabel("Recordings save directory:"), 2, 1, Qt.AlignRight)
        menu_layout.addWidget(self.file_path_widget)
        menu_layout.addWidget(save_button, 2, 3)
        menu_layout.addWidget(self.record_button, 3, 2)
        menu_layout.addWidget(QLabel("Model:"), 4, 1, Qt.AlignRight)
        menu_layout.addWidget(model_selection, 4, 2)
        menu_layout.addWidget(QLabel("Weights:"), 5, 1, Qt.AlignRight)
        menu_layout.addWidget(self.weights_selection, 5, 2)
        menu_layout.addWidget(self.predict_button, 6, 2)
        menu_layout.addWidget(QWidget())
        menu_widget.setLayout(menu_layout)

        keys_screen_splitter = QSplitter(Qt.Horizontal)
        keys_screen_splitter.addWidget(self.record_screen_label)
        keys_screen_splitter.addWidget(output_keys_widget)
        keys_screen_splitter.setSizes([400,200])
        keys_screen_splitter.setCollapsible(1, False)

        main_layout = QVBoxLayout()
        main_layout.addWidget(keys_screen_splitter)
        main_layout.addWidget(menu_widget)
        self.tab_record.setLayout(main_layout)

    def init_input_selection(self):
        input_selection = QComboBox()
        input_selection.activated.connect(self.select_input_source)
        joystick_count = pygame.joystick.get_count()
        for i in range(joystick_count):
            joystick = pygame.joystick.Joystick(i)
            input_selection.addItem("{}: {}".format(i, joystick.get_name()))
        if joystick_count == 0:
            input_selection.addItem("No connected devices found")
            input_selection.setEnabled(False)
            self.record_button.setEnabled(False)
        elif joystick_count == 1:
            self.select_input_source(0)
        return input_selection

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
        self.model = module.Model()
        self.model.create_model() 
        self.refresh_weights_selection()

    def refresh_weights_selection(self):
        self.weights_selection.setEnabled(True)
        self.available_weights = []
        weights_path = self.model_dir / "weights"
        self.weights_selection.clear()
        for weights in weights_path.iterdir():
            self.weights_selection.addItem(weights.name)
            self.available_weights.append(weights)
        if self.available_weights:
            self.select_weights(0)

    def select_weights(self, idx):
        self.model.model.load_weights(self.available_weights[idx])

    def select_input_source(self, idx):
        self.input_source = pygame.joystick.Joystick(idx)
        key_labels, key_events = recorder.capture_gamepad()
        self.key_labels = [PYGAME_TO_XBOX[label] for label in key_labels]

        self.key_event_widgets = []
        for idx, label in enumerate(self.key_labels):
            key_event_widget = QLabel(str(key_events[idx]))
            self.key_event_widgets.append(key_event_widget)
            self.output_keys_layout.addWidget(QLabel(label), idx, 1)
            self.output_keys_layout.addWidget(key_event_widget, idx, 2)

    def get_save_dir(self):
        save_dir_str = QFileDialog.getExistingDirectory(self, 'Select save directory')
        self.file_path_widget.setText(save_dir_str)
        self.save_dir = Path(save_dir_str) 

    def toggle_record_button(self):
        if self.record_button.isChecked():
            self.record_button.setStyleSheet("background-color: red")
            self.start_recording()
        else:
            self.record_button.setStyleSheet("")
            self.stop_recording()

    def toggle_predict_button(self):
        if self.predict_button.isChecked():
            self.predict_button.setStyleSheet("background-color: green")
            self.start_predicting()
        else:
            self.record_button.setStyleSheet("")
            self.stop_predicting()

    def start_predicting(self):
        self.record_button.setEnabled(False)
        self.predicting = True
        self.controller = PyvJoyXboxController(self.info["key_labels"])

    def stop_prediction(self):
        self.record_button.setEnabled(True)
        self.predicting = False
        self.controller = None

    def take_action(self, img):
        actions = self.model.get_actions(img)
        self.controller.emit_keys(actions)
        
    
    def stop_recording(self):
        self.recording = False
        self.predict_button.setEnabled(True)

    def start_recording(self):
        self.predict_button.setEnabled(False)
        self.frame = 1
        try_make_dirs(self.save_dir / "images")
        try_make_dirs(self.save_dir / "key-events")
        info_dict = {
                "key_labels": self.key_labels
                    }
        with open(self.save_dir / "info.json", "w") as fp:
            json.dump(info_dict, fp)
        self.recording = True

    
    def save_frame(self, image, key_events):
        np.save(self.save_dir / "image_{}".format(self.frame), image)
        np.save(self.save_dir / "key-event_{}".format(self.frame), key_events)

    def init_record_loop(self):
        self.updater = QTimer()
        self.updater.setSingleShot(True)
        self.updater.setInterval(100)
        self.updater.timeout.connect(self.record_frame)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close()

    def record_frame(self):
        self.updater.start()
        key_events = self.record_key_events()
        img = self.record_screen()
        if self.recording:
            self.save_frame(img, key_events)
            self.frame += 1
        elif self.predicting:
            self.take_action(img)
    
    def record_key_events(self):
        if not self.input_source:
            return
        key_labels, key_events = recorder.capture_gamepad()
        for idx, key_event_widget in enumerate(self.key_event_widgets):
            key_event_widget.setText("{0:.3f}".format(key_events[idx]))
        return key_events

    def record_screen(self):
        array_img = resize(recorder.capture_screen(), (66, 200), mode="reflect")
        array_img = 255 * array_img
        array_img = array_img.astype(np.uint8)
        height, width, channel = array_img.shape
        bytes_per_line = 3 * width
        q_img = QImage(array_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        q_pixmap = QPixmap.fromImage(q_img).scaled(self.record_screen_label.width(), self.record_screen_label.height(), Qt.KeepAspectRatio)
        self.record_screen_label.setPixmap(q_pixmap)
        self.record_screen_label.show()
        return array_img
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    Gui = Window()
    sys.exit(app.exec_())
