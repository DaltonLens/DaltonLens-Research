from cProfile import label
from dlcharts.common.utils import *
from dlcharts.common.dataset import LabeledImage

import cv2
import numpy as np
from zv.log import zvlog

import json
from pathlib import Path
from icecream import ic
from abc import ABC, abstractmethod

from dlcharts.pytorch import color_regression as cr

debug = False

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
            deltaS_100 *= target_hsv[1]
            deltaS_100 = max(deltaS_100, 1.0)
        else:
            deltaH_360 = deltaColorThreshold
            deltaS_100 = deltaColorThreshold
            deltaV_255 = deltaColorThreshold

        deltaS = deltaS_100 / 100.0
        return np.all(diff < np.array([deltaH_360, deltaS, deltaV_255]), axis=-1)

class DeepRegressionFinder(HSVFinder):
    def __init__(self, raw_image_rgb):
        processor = cr.Processor (Path(__file__).parent.parent.parent / "pretrained" / "regression_unetres_v1_scripted.pt")
        
        filtered_image_rgb = processor.process_image(raw_image_rgb)
        super().__init__(filtered_image_rgb, plot_mode=False)

def precision_recall_f1 (estimated_mask, gt_mask):
    num_gt_true = np.count_nonzero(gt_mask)
    num_estimated_true = np.count_nonzero(estimated_mask)
    correct_true = np.count_nonzero(estimated_mask & gt_mask)
    recall = np.float64(correct_true / num_gt_true)
    precision = correct_true / num_estimated_true
    f1_score = 2.0 * precision * recall / np.float64(precision + recall)
    if np.isnan(f1_score):
        f1_score = 0.0
    return (precision, recall, f1_score)

class InteractiveEvaluator:
    def __init__(self):
        pass

    def process_image (self, im_bgr: np.ndarray, finder: SimilarColorFinder, labeled_image: LabeledImage = None):
        def handle_click(event, x, y, flags, param):
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            mask = finder.similar_colors(x, y)
            zvlog.image ("estimated", bool_image_to_uint8(mask))
            # cv2.imshow ("estimated", bool_image_to_uint8(mask))

            if labeled_image is not None:
                label = labeled_image.labels_image[y,x]
                zvlog.image ("ground_truth", bool_image_to_uint8(labeled_image.mask_for_label(label)))
            
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("image", handle_click)
        cv2.imshow ("image", im_bgr)        
        while cv2.waitKey(0) != ord('q'):
            pass

def evaluate(labeled_image: LabeledImage, finder: SimilarColorFinder, easy_mode = False):
    # Fixed seed to make sure that we compare all the methods in
    # a deterministic way.
    from numpy.random import default_rng
    rng = default_rng(42)

    num_samples = 20
    labels = labeled_image.json['labels']
    labels = labels[1:] # remove the first label, it's the background
    num_samples_per_label = int(num_samples / len(labels))
    precision_recall_f1_scores = ([], [], [], [])
    wait_for_input = True
    for label_entry in labels:
        label = label_entry['label']
        label_rgb = np.array(label_entry['rgb_color'])
        mask_for_label = labeled_image.mask_for_label(label)
        rc_with_label = np.nonzero(mask_for_label)
        num_pixels_with_label = rc_with_label[0].size
        coordinates = rng.integers(0, num_pixels_with_label, num_samples_per_label)
        if debug:
            cv2.imshow ("gt_mask", mask_for_label.astype(np.uint8)*255)
        for coord in coordinates:
            r = rc_with_label[0][coord]
            c = rc_with_label[1][coord]

            if easy_mode:
                min_diff_rc = (r,c)
                min_diff = np.linalg.norm(labeled_image.rendered_image[r,c,:] - label_rgb)
                for dr, dc in  [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                    if not in_range(labeled_image.rendered_image, r+dr, c+dc):
                        continue
                    diff = np.linalg.norm(labeled_image.rendered_image[r+dr,c+dc,:] - label_rgb)
                    if (diff < min_diff):
                        min_diff = diff
                        min_diff_rc = (r+dr, c+dc)
                r, c = min_diff_rc

            estimated_mask = finder.similar_colors (c,r)

            precision, recall, f1 = precision_recall_f1 (estimated_mask, mask_for_label)
            precision_recall_f1_scores[0].append (precision)
            precision_recall_f1_scores[1].append (recall)
            precision_recall_f1_scores[2].append (f1)
            precision_recall_f1_scores[3].append ((r,c))
            if debug:
                ic (precision)
                ic (recall)
                zvlog.image ("estimated_mask", estimated_mask.astype(np.uint8)*255)
                if (wait_for_input and ic(cv2.waitKey(0)) == ord('q')):
                    wait_for_input = False
                else:
                    cv2.waitKey(1)
        
    average_precision = np.mean(precision_recall_f1_scores[0])
    average_recall = np.mean(precision_recall_f1_scores[1])
    average_f1 = np.mean(precision_recall_f1_scores[2])

    # Show all the values to get a quick overview.
    print ("(precision, recall, f1) = ", end =" ")
    for p,r,f1,rc in zip(*precision_recall_f1_scores):
        if p > 0.9 and r > 0.4:
            color_prefix, color_suffix = (TermColors.GREEN, TermColors.END)
        elif p > 0.8 and r > 0.3:
            color_prefix, color_suffix = (TermColors.YELLOW, TermColors.END)
        else:
            color_prefix, color_suffix = (TermColors.RED, TermColors.END)
            color_prefix += f"rc[{rc[0]},{rc[1]}]"
        print (f"{color_prefix}({p:.2f} {r:.2f} {f1:.2f}){color_suffix}", end =" ")
    print()

    num_good = np.count_nonzero (np.array(precision_recall_f1_scores[2]) >= 0.5)
    num_samples = len(precision_recall_f1_scores[0])
    percentage_good = 100.0*num_good/num_samples
    print (f"Average P,R,F1 over {num_samples} samples {average_precision:.2f}, {average_recall:.2f}, {average_f1:.2f}")
    print (f"Percentage of great results: {percentage_good:.2f}%")
    return (percentage_good, average_precision, average_recall, average_f1)

def main_interactive_evaluator():
    evaluator = InteractiveEvaluator()
    # labeled_image = LabeledImage (Path("generated/drawings-whitebg/img-00000-003.json"))
    labeled_image = LabeledImage (Path("inputs/opencv-generated/drawings-test/img-00000-000.json"))
    # labeled_image = LabeledImage (Path("generated/drawings/img-00000-001.json"))
    labeled_image.ensure_images_loaded()
    image_rgb = labeled_image.rendered_image
    # finder = HSVFinder(labeled_image.rendered_image, plot_mode=True)
    # labeled_image = None
    # image_rgb = swap_rb(cv2.imread("/home/nb/Perso/DaltonLens-Drive/Plots/Bowling.png", cv2.IMREAD_COLOR))
    finder = DeepRegressionFinder(image_rgb)
    evaluator.process_image (labeled_image.rendered_image, finder, labeled_image)

def main_batch_evaluation ():
    im = LabeledImage (Path("inputs/opencv-generated/drawings-test/img-00000-000.json"))
    # im = LabeledImage (Path("generated/drawings-whitebg/img-00000-003.json"))
    im.ensure_images_loaded()
    im.compute_labels_as_rgb()
    easy_mode = False
    # evaluate (im, HSVFinder(im.rendered_image, plot_mode=True), easy_mode=easy_mode)
    evaluate (im, DeepRegressionFinder(im.rendered_image), easy_mode=easy_mode)

def main():
    zvlog.start ()
    # main_batch_evaluation ()
    main_interactive_evaluator()

if __name__ == "__main__":
    main ()
