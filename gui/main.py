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
from PyQt5.QtGui import (QPixmap, QImage, QIntValidator)
from PyQt5.QtCore import (Qt, QTimer)

import recorder
from utilities import try_make_dirs
from gui.record_tab import RecordTab
from gui.process_tab import ProcessTab

class MainWindow(QTabWidget):

    def __init__(self):
        super().__init__()

        self.tab_record = RecordTab()
        self.tab_process = ProcessTab()

        self.addTab(self.tab_record, "Record/Predict")
        self.addTab(self.tab_process, "Process")
        
        self.setGeometry(1000, 0, 1000, 800)
        self.setMinimumWidth(300)
        self.setMinimumHeight(500)
        self.setWindowTitle('Project Kodo')
        self.show()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close()

def run_gui():        
    app = QApplication(sys.argv)
    Gui = MainWindow()
    sys.exit(app.exec_())
