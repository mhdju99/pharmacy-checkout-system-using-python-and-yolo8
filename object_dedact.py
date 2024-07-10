from ultralytics import YOLO
import cv2
from numpy import asarray
import cv2
import numpy as np
from matplotlib import pyplot as plt
from image_process import *

def prdict(model, img_recized):
    # pretrained YOLOv8n model
    results = model.predict(
        source=img_recized,
        show=False,
        save=False,
        conf=0.7,
        verbose=False,
        imgsz=640,
        show_labels=False,
        show_conf=False,
    )

    return results


