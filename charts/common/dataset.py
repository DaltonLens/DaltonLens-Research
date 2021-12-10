from charts.common.utils import *

import cv2
import numpy as np

import json
from pathlib import Path

class LabeledImage:
    def __init__(self, json_file: Path):
        self.json_file = json_file
        with open(json_file, 'r') as f:
            self.json = json.load(f)
        self.rendered_file = json_file.with_suffix('.rendered.png')
        self.labels_file = json_file.with_suffix('.labels.png')
        self.labels_image = None

    def release_images(self):
        self.labels_image = None
        self.rendered_image = None
        self.rendered_image_bgr = None

    def ensure_images_loaded(self):
        if self.labels_image is not None:
            return

        self.labels_image = cv2.imread(str(self.labels_file), cv2.IMREAD_GRAYSCALE)
        if debug:
            cv2.imshow (winname="label", mat=self.labels_image)
            cv2.waitKey (0)
        self.rendered_image_bgr = cv2.imread(str(self.rendered_file), cv2.IMREAD_COLOR)
        self.rendered_image = swap_rb(self.rendered_image_bgr)

    def mask_for_label(self, label: int):
        self.ensure_images_loaded()
        return self.labels_image == label

    def __repr__(self):
        return f"{self.json_file.name}"
