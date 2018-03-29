import sys
import os
import json
import importlib
import shutil
from pathlib import Path
import keyboard

import numpy as np
import time
import pygame
from cv2 import resize
from skimage.filters import gaussian
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout,
                             QHBoxLayout, QWidget, QComboBox, QPushButton, QSplitter, QFrame,
                             QTextEdit, QTabWidget, QFileDialog, QGridLayout, QLineEdit)
from PyQt5.QtGui import (QPixmap, QImage, QIntValidator)
from PyQt5.QtCore import (Qt, QTimer)

import recorder
from utilities import (try_make_dirs, sorted_alphanumeric)
# PyvJoy only works on Windows
if os.name == 'nt':
    from controllers import PyvJoyXboxController
from controllers import KeyboardController

class RecordTab(QWidget):

    def __init__(self):
        super().__init__()
        self.refresh_image = True
        self.recording = False
        self.predicting = False
        self.frame = 1
        self.interval = 25
        self.crop_first_secs = 2
        self.crop_last_secs = 2
        self.available_input_devices = []
        self.available_models = []
        self.model_dir = None
        self.available_weights = None 
        self.weights = None
        self.controller = None
        self.record_h = 66
        self.record_w = 200
        self.capture_h = 600
        self.capture_w = 800

        self.prev_time = time.time()

        self.init_record_UI()
        self.init_record_loop()

        self.updater.start()

    def init_record_UI(self):
        self.save_dir = None
        self.input_source = None

        self.record_button = QPushButton("Record")
        self.record_button.setEnabled(False)
        self.predict_button = QPushButton("Predict")

        keyboard.add_hotkey('ctrl+r', self.record_button.click)

        output_keys_widget = QWidget()
        self.output_keys_layout = QGridLayout()
        output_keys_widget.setLayout(self.output_keys_layout)

        menu_widget = QWidget()
        record_w_widget = QLineEdit(str(self.record_w))
        record_w_widget.setValidator(QIntValidator())
        record_w_widget.textChanged.connect(lambda new_val: self.line_edit_value_changed(new_val, "record_w"))
        record_h_widget = QLineEdit(str(self.record_h))
        record_h_widget.setValidator(QIntValidator())
        record_h_widget.textChanged.connect(lambda new_val: self.line_edit_value_changed(new_val, "record_h"))

        capture_w_widget = QLineEdit(str(self.capture_w))
        capture_w_widget.setValidator(QIntValidator())
        capture_w_widget.textChanged.connect(lambda new_val: self.line_edit_value_changed(new_val, "capture_w"))
        capture_h_widget = QLineEdit(str(self.capture_h))
        capture_h_widget.setValidator(QIntValidator())
        capture_h_widget.textChanged.connect(lambda new_val: self.line_edit_value_changed(new_val, "capture_h"))

        self.record_screen_label = QLabel(self)
        self.weights_selection = QComboBox()
        self.weights_selection.setEnabled(False)
        self.weights_selection.activated.connect(self.select_weights)
        input_selection = self.init_input_selection()
        model_selection = self.init_model_selection()
        self.file_path_widget = QLineEdit()
        self.file_path_widget.textChanged.connect(self.path_changed)
        self.record_button.toggle()
        self.record_button.setCheckable(True)
        self.record_button.clicked.connect(self.toggle_record_button)
        self.predict_button.toggle()
        self.predict_button.setCheckable(True)
        self.predict_button.clicked.connect(self.toggle_predict_button)

        resolution_widget = QWidget()
        resolution_layout = QGridLayout()
        resolution_layout.addWidget(record_w_widget, 1, 1, Qt.AlignRight)
        resolution_layout.addWidget(record_h_widget, 1, 2, Qt.AlignLeft)
        resolution_widget.setLayout(resolution_layout)

        capture_size_widget = QWidget()
        capture_size_layout = QGridLayout()
        capture_size_layout.addWidget(capture_w_widget, 1, 1, Qt.AlignRight)
        capture_size_layout.addWidget(capture_h_widget, 1, 2, Qt.AlignLeft)
        capture_size_widget.setLayout(capture_size_layout)

        menu_layout = QGridLayout()
        menu_layout.setColumnStretch(1, 1)
        menu_layout.setColumnStretch(2, 3)
        menu_layout.setColumnStretch(3, 1)
        menu_layout.addWidget(QWidget())
        menu_layout.addWidget(QLabel("Capture screen size"), 1, 1, Qt.AlignRight)
        menu_layout.addWidget(capture_size_widget, 1, 2)
        menu_layout.addWidget(QLabel("Recording resolution"), 2, 1, Qt.AlignRight)
        menu_layout.addWidget(resolution_widget, 2, 2)
        menu_layout.addWidget(QLabel("Input device:"), 3, 1, Qt.AlignRight)
        menu_layout.addWidget(input_selection, 3, 2)
        menu_layout.addWidget(QLabel("Data name:"), 4, 1, Qt.AlignRight)
        menu_layout.addWidget(self.file_path_widget)
        menu_layout.addWidget(self.record_button, 5, 2)
        menu_layout.addWidget(QLabel("Model:"), 6, 1, Qt.AlignRight)
        menu_layout.addWidget(model_selection, 6, 2)
        menu_layout.addWidget(QLabel("Weights:"), 7, 1, Qt.AlignRight)
        menu_layout.addWidget(self.weights_selection, 7, 2)
        menu_layout.addWidget(self.predict_button, 8, 2)
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
        self.setLayout(main_layout)

    def path_changed(self, new_path):
        if new_path != "":
            self.record_button.setEnabled(True)
        else:
            self.record_button.setEnabled(False)

    def line_edit_value_changed(self, new_val, variable_name):
        try:
            # Segmentation fault if screen_capture_size set to 1
            if int(new_val) > 1:
                setattr(self, variable_name, int(new_val))
        except ValueError:
            pass

    def init_input_selection(self):
        input_selection = QComboBox()
        input_selection.activated.connect(self.select_input_source)
        input_selection.addItem("0: Keyboard")
        self.available_input_devices.append(recorder.KeyboardRecorder())
        pygame.init()
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        for i in range(joystick_count):
            joystick = pygame.joystick.Joystick(i)
            input_selection.addItem("{}: {}".format(i + 1, joystick.get_name()))
            self.available_input_devices.append(recorder.GamepadRecorder(i))
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
        self.model = module.KodoModel()
        self.refresh_weights_selection()

    def refresh_weights_selection(self):
        self.weights_selection.setEnabled(True)
        self.available_weights = []
        weights_path = self.model_dir / "weights"
        self.weights_selection.clear()
        for weights in weights_path.iterdir():
            if not weights.is_dir() or weights.name == "__pycache__":
                continue
            self.weights_selection.addItem(weights.name)
            self.available_weights.append(weights)
        if self.available_weights:
            self.select_weights(0)

    def select_weights(self, idx):
        self.model.load_info(self.available_weights[idx] / "info.json")
        self.model.create_model(dropout_probability=0)
        self.model.model.load_weights(self.available_weights[idx] / "weights.h5")

    def select_input_source(self, idx):
        self.input_source = self.available_input_devices[idx]
        self.input_source.initialize_capture()
        self.key_labels, key_events = self.input_source.capture_events()

        self.key_event_widgets = []
        for idx, label in enumerate(self.key_labels):
            key_event_widget = QLabel(str(key_events[idx]))
            self.key_event_widgets.append(key_event_widget)
            self.output_keys_layout.addWidget(QLabel(label), idx, 1)
            self.output_keys_layout.addWidget(key_event_widget, idx, 2)

    def set_save_dir(self):
        new_name = self.file_path_widget.text()
        data_path = Path(os.getcwd()) / "data"
        current_data_names = os.listdir(data_path)
        idx = 1
        base_name = new_name
        while new_name in current_data_names:
            new_name = "{}_{}".format(base_name, idx)
            idx += 1
        self.save_dir = data_path / new_name

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
        self.refresh_image = False
        self.record_button.setEnabled(False)
        self.controller = KeyboardController(self.model.info["key_labels"])
        self.controller.reset_controller()
        self.predicting = True
    
    def stop_predicting(self):
        self.refresh_image = True
        self.record_button.setEnabled(True)
        self.predicting = False
        self.controller = None

    def take_action(self, img):
        actions = self.model.get_actions(img)
        self.controller.emit_keys(actions)
        
    
    def stop_recording(self):
        self.refresh_image = True
        self.recording = False
        self.predict_button.setEnabled(True)
        self.crop_ends()

    def start_recording(self):
        self.set_save_dir()
        self.refresh_image = False
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
        np.save(self.save_dir / "images" / "image_{}".format(self.frame), image)
        np.save(self.save_dir / "key-events" / "key-event_{}".format(self.frame), key_events)

    def init_record_loop(self):
        self.updater = QTimer()
        self.updater.setSingleShot(True)
        self.updater.setInterval(self.interval)
        self.updater.timeout.connect(self.record_frame)

    def record_frame(self):
        self.updater.start()
        current_time = time.time()
        #print(current_time - self.prev_time)
        self.prev_time = current_time
        if not self.predicting:
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
        key_labels, key_events = self.input_source.capture_events()
        for idx, key_event_widget in enumerate(self.key_event_widgets):
            try:
                key_event_widget.setText("{0:.3f}".format(key_events[idx]))
            except IndexError:
                print(idx, key_events)
        return key_events

    def crop_ends(self):
        crop_first_n_frames = int(self.crop_first_secs / (self.interval / 1000))
        crop_last_n_frames = int(self.crop_last_secs / (self.interval / 1000))
        
        n_frames = len(os.listdir(self.save_dir / "images"))
        if n_frames <= crop_first_n_frames + crop_last_n_frames + 10:
            shutil.rmtree(self.save_dir, ignore_errors=True)
        else:
            for idx in range(crop_first_n_frames):
                self.crop_frame(idx+1)
            for idx in range(crop_last_n_frames):
                self.crop_frame(self.frame-idx-1) 

    def crop_frame(self, frame_idx):
        os.remove(self.save_dir / "images" / "image_{}.npy".format(frame_idx))
        os.remove(self.save_dir / "key-events" / "key-event_{}.npy".format(frame_idx))

    def record_screen(self):
        array_img = recorder.capture_screen(capture_screen_width=self.capture_w, 
                                                capture_screen_height=self.capture_h)
        array_img = resize(array_img, (self.record_w, self.record_h))
        if self.refresh_image:
            height, width, channel = array_img.shape
            bytes_per_line = 3 * width
            q_img = QImage(array_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
            q_pixmap = QPixmap.fromImage(q_img).scaled(self.record_screen_label.width(), 
                                                       self.record_screen_label.height(), 
                                                       Qt.KeepAspectRatio)
            self.record_screen_label.setPixmap(q_pixmap)
            self.record_screen_label.show()
        return array_img

