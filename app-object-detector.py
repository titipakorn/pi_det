import time
import numpy as np
import cv2 as cv

import altusi.config as cfg
import altusi.visualizer as vis
from altusi import imgproc, helper
from altusi.logger import Logger

from altusi.objectdetector import ObjectDetector

from altusi.videos import WebcamVideoStream

import json

import requests
from io import BytesIO

# import logging

# # These two lines enable debugging at httplib level (requests->urllib3->http.client)
# # You will see the REQUEST, including HEADERS and DATA, and RESPONSE with HEADERS but without DATA.
# # The only thing missing will be the response.body which is not logged.
# try:
#     import http.client as http_client
# except ImportError:
#     # Python 2
#     import httplib as http_client
# http_client.HTTPConnection.debuglevel = 1

# # You must initialize logging, otherwise you'll not see debug output.
# logging.basicConfig()
# logging.getLogger().setLevel(logging.DEBUG)
# requests_log = logging.getLogger("requests.packages.urllib3")
# requests_log.setLevel(logging.DEBUG)
# requests_log.propagate = True

LOG = Logger('app-face-detector')

URL = "http://52.74.221.188/backend/track/"


def app(video_link, video_name, show, record, flip_hor, flip_ver):
    # initialize Face Detection net
    object_detector = ObjectDetector()

    # initialize Video Capturer
    #cap = cv.VideoCapture(video_link)
    cap = WebcamVideoStream(
        src=cfg.RTSP_URL).start()
    # (W, H), FPS = imgproc.cameraCalibrate(cap.stream)
    # LOG.info('Camera Info: ({}, {}) - {:.3f}'.format(W, H, FPS))

    if record:
        time_str = time.strftime(cfg.TIME_FM)
        writer = cv.VideoWriter(video_name+time_str+'.avi',
                                cv.VideoWriter_fourcc(*'XVID'), 20, (1280, 720))

    #cnt_frm = 0
    while True:
        frm = cap.read()
        if frm is None:
            continue
        #cnt_frm += 1
        # if(cnt_frm % 5 != 0):
        #     continue
        if flip_ver:
            frm = cv.flip(frm, 0)
        if flip_hor:
            frm = cv.flip(frm, 1)
        frm = imgproc.resizeByHeight(frm, 720)

        _start_t = time.time()
        images, bboxes = object_detector.getObjects(frm, def_score=0.5)
        if images:
            files = []
            for im in images:
                byte_io = BytesIO()
                im.save(byte_io, 'png')
                byte_io.seek(0)
                files.append(('files', byte_io))
            files.append(('bboxes', (None, json.dumps(
                bboxes), 'application/json')))
            if(len(bboxes)):
                requests.post(
                    url=URL, files=files)

        _prx_t = time.time() - _start_t

        if len(bboxes):
            frm = vis.plotBBoxes(frm, bboxes, len(bboxes) * ['person'])
        frm = vis.plotInfo(frm, 'Raspberry Pi - FPS: {:.3f}'.format(1/_prx_t))
        frm = cv.cvtColor(np.asarray(frm), cv.COLOR_BGR2RGB)

        if record:
            writer.write(frm)

        # if show:
        #     cv.imshow(video_name, frm)
        #     key = cv.waitKey(1)
        #     if key in [27, ord('q')]:
        #         LOG.info('Interrupted by Users')
        #         break

    if record:
        writer.release()
    # cap.release()
    # cv.destroyAllWindows()


def main(args):
    video_link = args.video if args.video else 0
    app(video_link, args.name, args.show,
        args.record, args.flip_hor, args.flip_ver)


if __name__ == '__main__':
    LOG.info('Raspberry Pi: Object Detection')

    args = helper.getArgs()
    main(args)

    LOG.info('Process done')
