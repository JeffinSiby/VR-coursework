import cv2
import numpy as np

live_window = "bufferDisplay"
cv2.namedWindow(live_window, cv2.WINDOW_NORMAL)

WAIT_TIME = int((1/256)*512)
frame_arr_dump = np.load("./frame_buffer_output.npz")
frame_arr = frame_arr_dump["data"]
for index, frame in enumerate(frame_arr):
    cv2.imshow(live_window, frame)
    cv2.waitKey(WAIT_TIME)