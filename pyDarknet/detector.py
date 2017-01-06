from libpydarknet import DarknetObjectDetector

from PIL import Image
import numpy as np
import time

print 'succc'

class DetBBox(object):

    def __init__(self, bbox):
        self.left = bbox.left
        self.right = bbox.right
        self.top = bbox.top
        self.bottom = bbox.bottom
        self.confidence = bbox.confidence
        self.cls = bbox.cls

class Darknet_ObjectDetector():

    def __init__(self, net_cfg, weight):
        self._detector = DarknetObjectDetector(net_cfg, weight)

    def detect_object(self, im):
        start = time.time()

        data = np.array(im).transpose([2,0,1]).astype(np.uint8)

        rst = self._detector.detect_object(data.tostring(), im.shape[1], im.shape[0], 3)

        end = time.time()

        ret_rst = [DetBBox(x) for x in rst]

        return ret_rst, end-start

    @staticmethod
    def set_device(gpu_id):
        DarknetObjectDetector.set_device(gpu_id)
