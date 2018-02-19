from pathlib import Path
import json

import numpy as np 

class KodoModel:
    
    def __init__(self):
        self.X = None
        self.y = None
        self.info = None

    def process(self):
        raise NotImplementedError
    
    def create_model(self):
        raise NotImplementedError

    def load_data(self, data_folder_str):
        self.data_folder_name = data_folder_str.split("/")[-1]
        data_folder = Path(data_folder_str)
        self.y = np.load(data_folder / "y.npy")
        self.X = np.load(data_folder / "X.npy")
        self.info = json.load(data_folder / "info.json")

    def load_info(self, info_path):
        with open(info_path) as info_file:
            self.info = json.load(info_file)

    def train(self):
        raise NotImplementedError

    def get_actions(self):
        raise NotImplementedError

