from detector import Darknet_ObjectDetector as ObjectDetector
from detector import DetBBox

import requests
from PIL import Image
from PIL import ImageFilter
from StringIO import StringIO
import cv2
def _get_image(url):
    return Image.open(StringIO(requests.get(url).content))
