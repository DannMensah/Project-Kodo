import numpy as np
import mss
import pygame as pg
import keyboard
from controller_mappings import PYGAME_TO_XBOX

# Captures part of the screen and returns the resulting pixels as a NumPy array
def capture_screen(capture_screen_x=0, capture_screen_y=40, capture_screen_width=800, capture_screen_height=600):
    # Part of the screen to capture
    crop = {'top': capture_screen_y, 
               'left': capture_screen_x, 
               'width': capture_screen_width, 
               'height': capture_screen_height}
    # Save pixels from the screen to a NumPy array
    img_bgra = np.array(mss.mss().grab(crop))
    B, G, R, A = np.rollaxis(img_bgra,2)
    img = np.stack([R, G, B], axis=2)
    return img

class GamepadRecorder:

    def __init__(self, gamepad_id=0):
        self.gamepad_id = gamepad_id

    def initialize_capture(self):
        self.joystick = pg.joystick.Joystick(self.gamepad_id)
        self.joystick.init()

    def capture_events(self):
        pg.event.get()
        
        key_events = []
        key_labels = []

        n_axes = self.joystick.get_numaxes()
        axes = [self.joystick.get_axis(i) for i in range(n_axes)]
        key_events += axes
        key_labels += ["axis_{}".format(i) for i in range(n_axes)]
                
        n_buttons = self.joystick.get_numbuttons()
        buttons = [self.joystick.get_button(i) for i in range(n_buttons)] 
        key_events += buttons
        key_labels += ["button_{}".format(i) for i in range(n_buttons)]

        n_hats = self.joystick.get_numhats()
        hats = [self.joystick.get_hat(i) for hat in range(n_hats)]
        key_events += hats
        key_labels += ["hat_{}".append(i) for i in range(n_hats)]
        
        key_labels = [PYGAME_TO_XBOX[label] for label in key_labels]


        return (key_labels, key_events)

    def deactivate(self):
        pg.quit()


class KeyboardRecorder:
    def __init__(self):
        pass

    def initialize_capture(self):
        pass

    def capture_events(self):
        captured_keys = ["up", "down", "left", "right"]
        key_states = []
        no_op = True
        for key in captured_keys:
            if keyboard.is_pressed(key):
                key_states.append(1)
                no_op = False
            else:
                key_states.append(0)
        if no_op:
            key_states.append(1)
        else:
            key_states.append(0)
        captured_keys.append("no_op")
        return (captured_keys, key_states)

    def deactivate(self):
        pass
