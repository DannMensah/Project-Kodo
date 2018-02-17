import sys
import numpy as np
import time
import pygame
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QComboBox, QPushButton, QSplitter, QFrame, QTextEdit, QTabWidget, QFileDialog, QGridLayout)
from PyQt5.QtGui import (QPixmap, QImage)
from PyQt5.QtCore import (Qt, QTimer)

import record

class Window(QTabWidget):

    def __init__(self):
        super().__init__()
        
        self.tab_record = QWidget()
        self.tab_process = QWidget()
        self.tab_predict = QWidget()

        self.addTab(self.tab_record, "Record")
        self.addTab(self.tab_process, "Process")
        self.addTab(self.tab_predict, "Predict")
        
        self.init_record_UI()

        self.setGeometry(1000, 0, 1000, 800)
        self.setMinimumWidth(300)
        self.setMinimumHeight(500)
        self.setWindowTitle('pyqt5 Tut')
        # self.setWindowIcon(QIcon('pic.png'))
        self.init_record_loop()
        self.show()
        self.updater.start()

    def init_record_UI(self):
        self.save_dir = None
        self.input_source = None

        record.init_gamepad_capture()

        output_keys_widget = QWidget()
        self.output_keys_layout = QGridLayout()
        output_keys_widget.setLayout(self.output_keys_layout)

        menu_widget = QWidget()
        self.record_screen_label = QLabel(self)
        input_selection = self.init_input_selection()
        save_button = QPushButton("Choose...")
        save_button.clicked.connect(self.get_save_dir)

        menu_layout = QGridLayout()
        menu_layout.setColumnStretch(1, 1)
        menu_layout.setColumnStretch(2, 3)
        menu_layout.addWidget(QWidget())
        menu_layout.addWidget(QLabel("Input device:"), 1, 1, Qt.AlignRight)
        menu_layout.addWidget(input_selection, 1, 2)
        menu_layout.addWidget(QLabel("Recordings save directory:"), 2, 1, Qt.AlignRight)
        menu_layout.addWidget(save_button, 2, 2)
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
        elif joystick_count == 1:
            self.select_input_source(0)
        return input_selection

    def select_input_source(self, idx):
        self.input_source = pygame.joystick.Joystick(idx)
        key_labels, key_events = record.capture_gamepad()
        self.key_event_widgets = []
        for idx, label in enumerate(key_labels):
            key_event_widget = QLabel(str(key_events[idx]))
            self.key_event_widgets.append(key_event_widget)
            self.output_keys_layout.addWidget(QLabel(label), idx, 1)
            self.output_keys_layout.addWidget(key_event_widget, idx, 2)

    def get_save_dir(self):
        self.save_dir = QFileDialog.getExistingDirectory(self, 'Select save directory')
        print(self.save_dir)

    def init_record_loop(self):
        self.updater = QTimer()
        self.updater.setSingleShot(True)
        self.updater.setInterval(100)
        self.updater.timeout.connect(self.record_frame)


    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close()

    def record_frame(self):
        self.record_key_events()
        self.record_screen()
        self.updater.start()
    
    def record_key_events(self):
        if not self.input_source:
            return
        key_labels, key_events = record.capture_gamepad()
        for idx, key_event_widget in enumerate(self.key_event_widgets):
            key_event_widget.setText("{0:.3f}".format(key_events[idx]))

    def record_screen(self):
        array_img = record.capture_screen()
        height, width, channel = array_img.shape
        bytes_per_line = 3 * width
        q_img = QImage(array_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        q_pixmap = QPixmap.fromImage(q_img).scaled(self.record_screen_label.width(), self.record_screen_label.height(), Qt.KeepAspectRatio)
        self.record_screen_label.setPixmap(q_pixmap)
        self.record_screen_label.show()
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    Gui = Window()
    sys.exit(app.exec_())
