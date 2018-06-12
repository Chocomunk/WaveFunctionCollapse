import cv2
import os
import numpy as np


def load_tiles(dirname):
    tiles = []
    directory = os.fsencode(dirname)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image = cv2.imread(os.path.join(dirname, filename))
            if image is not None:
                tiles.append(image)
    return tiles
