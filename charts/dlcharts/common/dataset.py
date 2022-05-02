from .utils import debug, swap_rb
from daltonlens import convert

from zv.log import zvlog

import cv2
import numpy as np

import json
from pathlib import Path

def Lab_from_sRGB(im_srgb_uint8: np.ndarray):
    """Convert an sRGB image to CIE L*a*b
    
    The input sRGB range is expected to be in [0,255]

    The output L*a*b* range is [0,100] for L, and [-127,127] for a and b.
    """
    linearRGB = convert.linearRGB_from_sRGB (convert.as_float32(im_srgb_uint8))
    xyz = convert.apply_color_matrix(linearRGB, convert.XYZ_from_linearRGB_BT709)
    return convert.Lab_from_XYZ(xyz)
    
class LabeledImage:
    def __init__(self, json_file: Path):
        self.json_file = json_file
        with open(json_file, 'r') as f:
            self.json = json.load(f)
        self.rendered_file = json_file.with_suffix('.antialiased.png')
        self.labels_file = json_file.with_suffix('.labels.png')
        self.labels_image = None
        self.labels_as_rgb = None
        self.rendered_image = None
        self.labels_quantized_Lab = None

    def release_images(self):
        self.labels_image = None
        self.rendered_image = None
        self.labels_as_rgb = None
        self.labels_quantized_Lab = None

    def compute_quantized_Lab(self):
        if self.labels_quantized_Lab is not None:
            return
    
        self.ensure_images_loaded ()
        labels_lut = np.zeros((256,3), dtype=np.uint8)
        for label_entry in self.json['labels']:
            label: int = label_entry['label']
            rgb = np.array(label_entry['rgb_color'], dtype=np.uint8)
            # Create an image
            rgb = rgb[np.newaxis, np.newaxis, :]
            lab = Lab_from_sRGB (rgb).reshape(3)
            # L*a*b* range for L is [0,100] and [-128,127] for a and b.
            n_l_bins = 8
            n_ab_bins = 16
            quantized_lab = np.floor(np.array([
                n_l_bins * lab[0] / 100.01,
                n_ab_bins * (lab[1] + 128.0) / 255.01,
                n_ab_bins * (lab[2] + 128.0) / 255.01,
            ]))
            quantized_lab = quantized_lab.astype(dtype=np.uint8)
            labels_lut[label] = quantized_lab            

        assert np.max(labels_lut[...,0] < n_l_bins)
        assert np.max(labels_lut[...,1] < n_ab_bins)
        assert np.max(labels_lut[...,2] < n_ab_bins)
        self.labels_quantized_Lab = labels_lut[self.labels_image]

        if debug:
            zvlog.image ("label", self.labels_image)
            scaled_lab = np.around(np.multiply(self.labels_quantized_Lab, np.array([255/16, 255/8, 255/8]))).astype(np.uint8)
            zvlog.image ("labels_quantized_Lab", scaled_lab)

    def compute_labels_as_rgb(self):
        if self.labels_as_rgb is not None:
            return

        self.ensure_images_loaded()

        labels_lut = np.zeros((256,3), dtype=np.uint8)
        for label_entry in self.json['labels']:
            label: int = label_entry['label']
            rgb = np.array(label_entry['rgb_color'], dtype=np.uint8)
            labels_lut[label] = rgb

        self.labels_as_rgb = labels_lut[self.labels_image]
        
        # Grab the background from the original image.
        background_mask = np.all(self.labels_as_rgb == labels_lut[0], axis=-1)
        if debug:
            zvlog.image ("background_mask", background_mask.astype(np.uint8)*255)
            zvlog.image ("rendered_image", self.rendered_image)
        background_mask = np.expand_dims(background_mask, axis=-1)
        self.labels_as_rgb = np.where(background_mask, self.rendered_image, self.labels_as_rgb)

        if debug:
            zvlog.image ("label", self.labels_image)
            zvlog.image ("label_as_rgb", swap_rb(self.labels_as_rgb))

    def ensure_images_loaded(self):
        if self.labels_image is not None:
            assert self.rendered_image is not None
            return

        self.labels_image = cv2.imread(str(self.labels_file), cv2.IMREAD_GRAYSCALE)
        self.rendered_image = swap_rb(cv2.imread(str(self.rendered_file), cv2.IMREAD_COLOR))
        assert self.rendered_image is not None

    def mask_for_label(self, label: int):
        self.ensure_images_loaded()
        return self.labels_image == label

    def __repr__(self):
        return f"{self.json_file.name}"
