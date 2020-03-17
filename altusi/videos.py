# import the necessary packages
import datetime
from threading import Thread
from PIL import Image
import numpy as np
from io import BytesIO
import requests
import cv2


class WebcamVideoStream:
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame
        # from the stream

        self.URL = src
        r = requests.get(self.URL)
        i = Image.open(BytesIO(r.content))
        frame = np.array(i)
        self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
            try:
                # otherwise, read the next frame from the stream
                r = requests.get(self.URL)
                i = Image.open(BytesIO(r.content))
                frame = np.array(i)
                self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except:
                pass

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
