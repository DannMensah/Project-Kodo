from pathlib import Path
import json

import numpy as np 

class KodoTemplate:
    
    def __init__(self):
        self.X = None
        self.y = None
        self.info = None

    def process(self):
        raise NotImplementedError
    
    def create_model(self):
        raise NotImplementedError

    def load_info(self, info_path):
        with open(info_path) as info_file:
            self.info = json.load(info_file)

    def train(self):
        raise NotImplementedError

    def get_actions(self):
        raise NotImplementedError
