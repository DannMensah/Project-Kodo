import numpy as np
import mss
import pygame

# Captures part of the screen and returns the resulting pixels as a NumPy array
def capture_screen(capture_screen_x=0, capture_screen_y=40, capture_screen_width=800, capture_screen_height=640):
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

def init_gamepad_capture(gamepad_id=0):
    pygame.init()
    pygame.joystick.init()
    joystick_count = pygame.joystick.get_count()
    for i in range(joystick_count):
        joystick = pygame.joystick.Joystick(i)
        joystick.init()

def capture_gamepad(gamepad_id=0):
    pygame.event.get()
    joystick = pygame.joystick.Joystick(gamepad_id)

    name = joystick.get_name()

    n_axes = joystick.get_numaxes()
    axes = [joystick.get_axis(i) for i in range(n_axes)]

    for i in range( n_axes ):
            axis = joystick.get_axis( i )
            print(axis)

    n_buttons = joystick.get_numbuttons()
    buttons = [joystick.get_button(i) for i in range(n_buttons)] 

    n_hats = joystick.get_numhats() 
    hats = [joystick.get_hat(i) for hat in range(n_hats)]

    return (axes, buttons, hats)

def stop_gamepad_capture():
    pygame.quit()
