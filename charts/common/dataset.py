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
        self.labels_as_rgb = None
        self.rendered_image = None
        self.rendered_image_bgr = None

    def release_images(self):
        self.labels_image = None
        self.rendered_image = None
        self.rendered_image_bgr = None
        self.labels_as_rgb = None

    def ensure_images_loaded(self):
        if self.labels_image is not None:
            return

        self.labels_image = cv2.imread(str(self.labels_file), cv2.IMREAD_GRAYSCALE)
        labels_lut = np.zeros((256,3), dtype=np.uint8)
        for label_entry in self.json['labels']:
            label: int = label_entry['label']
            rgb = np.array(label_entry['rgb_color'], dtype=np.uint8)
            labels_lut[label] = rgb

        self.labels_as_rgb = labels_lut[self.labels_image]
        if debug:
            cv2.imshow ("label", self.labels_image)
            cv2.imshow ("label_as_rgb", swap_rb(self.labels_as_rgb))
            cv2.waitKey (0)
        self.rendered_image_bgr = cv2.imread(str(self.rendered_file), cv2.IMREAD_COLOR)
        self.rendered_image = swap_rb(self.rendered_image_bgr)
        assert self.rendered_image is not None

    def mask_for_label(self, label: int):
        self.ensure_images_loaded()
        return self.labels_image == label

    def __repr__(self):
        return f"{self.json_file.name}"