import pyvjoy

from utilities import XBOX_TO_PYVJOY

class PyvJoyXboxController:
    def __init__(self, data_key_labels, button_threshold=0.5):
        self.controller = pyvjoy.VJoyDevice(1)
        self.data_key_labels = data_key_labels
        self.button_threshold = button_threshold
    
    def scale_axis(self, value):
        max_vjoy = 32767
        # [-1, 1] to [0, 32767]
        return ((value + 1) / 2) * 32767

    def threshold_button(self, value):
        if value >= button_threshold:
            return 1
        return 0

    def emit_keys(output_values):
        for idx, value in enumerate(output_values):
            key_label = self.data_key_labels[i]
            if key_label in XBOX_TO_PYVJOY["AXES"].keys():
                scaled = self.scale_axis(value)
                j.set_axis(XBOX_TO_PYVJOY[key_label], scaled)
            elif key_label in XBOX_TO_PYVJOY["BUTTONS"].keys():
                thresholded = threshold_button(value)
                j.set_button(XBOX_TO_PYVJOY["BUTTONS"], thresholded)
