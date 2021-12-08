from pathlib import Path
import json
import cv2
import numpy as np

import sklearn.metrics

from icecream import ic

from abc import ABC, abstractmethod

debug = True

def swap_rb(im):
    return im[:,:,[2,1,0]]

class SimilarColorFinder(ABC):
    def __init__(self, image_rgb):
        self.float_image = image_rgb.astype(np.float32)

    # Baseline implemented with HSV
    @abstractmethod
    def similar_colors(self, c, r):
        return None

class HSVFinder(SimilarColorFinder):
    def __init__(self, image_rgb, plot_mode=True):
        super().__init__(image_rgb)
        self.plot_mode = plot_mode
        # H [0,360] S [0,1] V [0,255]
        self.hsv_image = cv2.cvtColor(swap_rb(self.float_image), cv2.COLOR_BGR2HSV)

    # HSV-based method implemented by DaltonLens-Desktop
    def similar_colors(self, c, r):
        target_hsv = self.hsv_image[r,c,:]
        target_rgb = self.float_image[r,c,:]
        
        if debug:
            ic(target_rgb)
            ic(target_hsv)
            ic((c,r))

        distance_from_target = np.linalg.norm(self.float_image - target_rgb, axis=-1)

        diff = np.abs(self.hsv_image - target_hsv)
        # h is modulo 360º
        diff[:,:,0] = np.minimum(diff[:,:,0], 360.0 - diff[:,:,0])

        # Adapted from DaltonLens-Desktop
        # The plot-mode has a much higher tolerance on value and saturation
        # because anti-aliased lines are blended with the background, which is
        # typically black/white/gray and will desaturate the line color after
        # alpha blending.
        # If the selected color is already not very saturated (like on aliased
        # an edge), then don't tolerate a huge delta.
        deltaColorThreshold = 10

        if self.plot_mode:
            deltaH_360 = deltaColorThreshold
            deltaS_100 = deltaColorThreshold * 5.0 # tolerates 3x more since the range is [0,100]
            deltaV_255 = deltaColorThreshold * 12.0 # tolerates much more difference than hue.
            deltaS_100 *= target_hsv[1] * 100.0
            deltaS_100 = max(deltaS_100, 1.0)
        else:
            deltaH_360 = deltaColorThreshold
            deltaS_100 = deltaColorThreshold
            deltaV_255 = deltaColorThreshold

        deltaS = deltaS_100 / 100.0
        return np.all(diff < np.array([deltaH_360, deltaS, deltaV_255]), axis=-1)

class LabeledImage:
    def __init__(self, json_file: Path):
        self.json_file = json_file
        with open(json_file, 'r') as f:
            self.json = json.load(f)
        print (self.json)
        self.rendered_file = json_file.with_suffix('.rendered.png')
        self.labels_file = json_file.with_suffix('.labels.png')
        self.labels_image = None

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
        

def evaluate(labeled_image: LabeledImage, finder: SimilarColorFinder):
    from numpy.random import default_rng
    
    num_samples = 100

    # Fixed seed to make sure that we compare all the methods in a deterministic
    # way.
    rng = default_rng(42)

    labels = labeled_image.json['labels']
    labels = labels[1:] # remove the first label, it's the background
    num_samples_per_label = int(num_samples / len(labels))
    precision_recall_scores = ([], [])
    wait_for_input = True
    for label_entry in labels:
        label = label_entry['label']
        mask_for_label = labeled_image.mask_for_label(label)
        rc_with_label = np.nonzero(mask_for_label)
        num_pixels_with_label = rc_with_label[0].size
        coordinates = rng.integers(0, num_pixels_with_label, num_samples_per_label)
        if debug:
            cv2.imshow ("gt_mask", mask_for_label.astype(np.uint8)*255)
        for coord in coordinates:
            r = rc_with_label[0][coord]
            c = rc_with_label[1][coord]
            estimated_mask = finder.similar_colors (c,r)
            precision = sklearn.metrics.precision_score (mask_for_label.flatten(), estimated_mask.flatten())
            recall = sklearn.metrics.recall_score (mask_for_label.flatten(), estimated_mask.flatten())
            precision_recall_scores[0].append (precision)
            precision_recall_scores[1].append (recall)
            if debug:
                ic (precision)
                ic (recall)
                cv2.imshow ("estimated_mask", estimated_mask.astype(np.uint8)*255)
                if (wait_for_input and ic(cv2.waitKey(0)) == ord('q')):
                    wait_for_input = False
                else:
                    cv2.waitKey(1)
        
    average_precision = np.mean(precision_recall_scores[0])
    average_recall = np.mean(precision_recall_scores[1])
    print (f"Average of precision scores over {len(precision_recall_scores[0])} samples: {average_precision}")
    print (f"   Average of recall scores over {len(precision_recall_scores[1])} samples: {average_recall}")
    return (average_precision, average_recall)

if __name__ == "__main__":
    # im = LabeledImage (Path("generated/drawings/img-00000-000.json"))
    im = LabeledImage (Path("generated/drawings-whitebg/img-00000-003.json"))
    im.ensure_images_loaded()
    evaluate (im, HSVFinder(im.rendered_image, plot_mode=True))

