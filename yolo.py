import sys
import glob
import os
from PIL import Image
import numpy as np
from cStringIO import StringIO
import pyDarknet
import base64
import cPickle
import time
import cv2

gpu = 0

cnt = 0

#init detector
pyDarknet.ObjectDetector.set_device(gpu)
detector = pyDarknet.ObjectDetector('cfg/tiny-yolo-voc.cfg', 'tiny-yolo-voc.weights')


for i in range(1000):
    im_data = 'data/dog.jpg'

    im_org = cv2.imread(im_data)
    im = cv2.cvtColor(im_org, cv2.COLOR_BGR2RGBA)

    rst, rt = detector.detect_object(im)

    print len(rst)
    for i in range(len(rst)):
        cv2.rectangle(im_org, (rst[i].left, rst[i].top), (rst[i].right, rst[i].bottom), (255, 0, 0))
    cv2.imshow('1', im_org)
    cv2.waitKey(10)
    cnt += 1

    print cnt, rt




