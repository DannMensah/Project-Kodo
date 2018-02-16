import numpy as np
import cv2
import mss
import mss.tools
import time
from utilities.controller import XboxController


def record_mss(frame_rate=10, capture_screen_width=800, capture_screen_height=640, output_screen_scale=0.5):
    controller = XboxController()
    time_per_frame = 1/frame_rate
    cv2.namedWindow('Recording',cv2.WINDOW_NORMAL)
    last_capture_time = time.time()
    with mss.mss() as sct:
        # Part of the screen to capture
        monitor = {'top': 40, 'left': 0, 'width': capture_screen_width, 'height': capture_screen_height}
        while 'Screen capturing':
            if time.time() - last_capture_time < time_per_frame:
                continue
            else:
                print(time.time() - last_capture_time)
                print(controller.read())
                last_capture_time = time.time()
            # Get raw pixels from the screen, save it to a Numpy array
            img = np.array(sct.grab(monitor))

            # Display the picture
            cv2.imshow("Recording", img)
            output_screen_width = int(capture_screen_width * output_screen_scale)
            output_screen_height = int(capture_screen_height * output_screen_scale)
            cv2.resizeWindow("Recording", output_screen_width, output_screen_height)
            # Press "q" to quit
            # print( 1 / (time.time() - last_capture_time))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
