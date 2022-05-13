from cProfile import label
from dlcharts.common.utils import *
from dlcharts.common.dataset import LabeledImage

import cv2
import numpy as np
from torch import per_channel_affine_float_qparams
import torch
from zv.log import zvlog
import zv

import json
from pathlib import Path
from icecream import ic
from abc import ABC, abstractmethod
import sys
import itertools
import shutil

from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple

from dlcharts.pytorch import color_regression as cr

import argparse

debug = False

class Rating(Enum):
    GOOD=0
    POOR=1
    BAD=2

class SimilarColorFinder(ABC):
    def __init__(self, image_rgb, raw_image_rgb = None):
        self.image_rgb = image_rgb
        self.float_image = image_rgb.astype(np.float32)
        self.raw_image_rgb = raw_image_rgb if raw_image_rgb is not None else image_rgb

    # Baseline implemented with HSV
    @abstractmethod
    def similar_colors(self, c, r):
        return None

class RGBFinder(SimilarColorFinder):
    def __init__(self, image_rgb, raw_image_rgb):
        super().__init__(image_rgb, raw_image_rgb)

    # Simple RGB method
    def similar_colors(self, c, r):
        target_rgb = self.float_image[r,c,:]

        rgbThreshold = 20.0
        diff = np.max(np.abs(self.float_image - target_rgb), axis=-1)
        return diff < rgbThreshold

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

        plot_mode_for_this_pixel = self.plot_mode
        if target_hsv[1] < 0.1:
            plot_mode_for_this_pixel = False

        # FIXME: comparing H sucks when saturation is low. DaltonLens has a hack
        # for this, but we should just use something else.
        if plot_mode_for_this_pixel:
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

class DeepRegressionFinder(RGBFinder):
    def __init__(self, raw_image_rgb, model):
        if isinstance(model, torch.nn.Module):
            # Interpret as a loaded model.
            processor = cr.Processor (model)
        else:
            # Interpret as a file.
            processor = cr.Processor (Path(__file__).parent.parent.parent / "pretrained" / model)
        
        filtered_image_rgb, maybe_cropped_raw_rgb = processor.process_image(raw_image_rgb)
        super().__init__(filtered_image_rgb, maybe_cropped_raw_rgb)

def precision_recall_f1 (estimated_mask, gt_mask):
    # Convert to float early to avoid exceptions because of a division by int 0
    num_gt_true = np.float64(np.count_nonzero(gt_mask))
    num_estimated_true = np.float64(np.count_nonzero(estimated_mask))
    correct_true = np.float64(np.count_nonzero(estimated_mask & gt_mask))
    recall = correct_true / num_gt_true
    precision = correct_true / num_estimated_true
    f1_score = 2.0 * precision * recall / np.float64(precision + recall)
    if np.isnan(f1_score):
        f1_score = 0.0
    return (precision, recall, f1_score)

class InteractiveEvaluator:
    def __init__(self):
        pass

    def process_image (self, im_rgb: np.ndarray, finder: SimilarColorFinder, labeled_image: LabeledImage = None):
        app = zv.App()
        app.initialize ()
        viewer = app.getViewer()
        viewer.setLayout (1, 3)

        def event_callback (image_id, x, y, user_data):
            if not zv.imgui.IsMouseClicked(zv.imgui.MouseButton.Left, False):
                return
            x,y = int(x), int(y)
            mask = finder.similar_colors(x, y)
            viewer.addImage ("estimated", bool_image_to_uint8(mask), replace=True)
            # cv2.imshow ("estimated", bool_image_to_uint8(mask))

            if labeled_image is not None:
                label = labeled_image.labels_image[y,x]
                viewer.addImage ("ground_truth", bool_image_to_uint8(labeled_image.mask_for_label(label)))
        
        image_id = viewer.addImage ("Input image", finder.raw_image_rgb)
        viewer.setEventCallback(image_id, event_callback, None)

        viewer.addImage("Filtered", finder.image_rgb)

        while app.numViewers > 0:
            app.updateOnce (1.0 / 30.0)

@dataclass
class Score:
    precision: float
    recall: float
    f1: float
    rating: Rating
    rc: Tuple

@dataclass
class Results:
    percentage_good: float
    average_precision: float
    average_recall: float
    average_f1: float

def compute_rating(p, r, f1):
    if p > 0.9 and r > 0.4:
        return Rating.GOOD
    elif p > 0.8 and r > 0.3:
        return Rating.POOR
    else:
        return Rating.BAD

def evaluate(labeled_image: LabeledImage, finder: SimilarColorFinder, easy_mode = False):
    # Fixed seed to make sure that we compare all the methods in
    # a deterministic way.
    from numpy.random import default_rng
    rng = default_rng(42)

    num_samples = 20
    labels = labeled_image.json['labels']
    labels = labels[1:] # remove the first label, it's the background
    num_samples_per_label = int(num_samples / len(labels))
    precision_recall_f1_scores: List[Score] = []
    wait_for_input = True
    for label_entry in labels:
        label = label_entry['label']
        label_rgb = np.array(label_entry['rgb_color'])
        mask_for_label = labeled_image.mask_for_label(label)
        rc_with_label = np.nonzero(mask_for_label)
        num_pixels_with_label = rc_with_label[0].size
        if num_pixels_with_label == 0:
            continue
        coordinates = rng.integers(0, num_pixels_with_label, num_samples_per_label)
        if debug:
            zvlog.image ("gt_mask", mask_for_label.astype(np.uint8)*255)
        for coord in coordinates:
            r = rc_with_label[0][coord]
            c = rc_with_label[1][coord]

            # Try to find a nearby pixel that is closer to the true label.
            # Makes sure it still has the right label though.
            if easy_mode:
                min_diff_rc = (r,c)
                min_diff = np.linalg.norm(labeled_image.rendered_image[r,c,:] - label_rgb)
                for dr, dc in  [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                    if not in_range(labeled_image.rendered_image, r+dr, c+dc):
                        continue
                    # Make sure it still has the right label. Sometimes with a very weak alpha
                    # the closest rgb might have a different label.
                    if not mask_for_label[r+dr, c+dc]:
                        continue
                    diff = np.linalg.norm(labeled_image.rendered_image[r+dr,c+dc,:] - label_rgb)
                    if (diff < min_diff):
                        min_diff = diff
                        min_diff_rc = (r+dr, c+dc)
                r, c = min_diff_rc

            estimated_mask = finder.similar_colors (c,r)

            precision, recall, f1 = precision_recall_f1 (estimated_mask, mask_for_label)
            score = Score(precision, recall, f1, compute_rating(precision, recall, f1), (r,c))
            precision_recall_f1_scores.append(score)
            if debug:
                ic (precision)
                ic (recall)
                zvlog.image ("estimated_mask", estimated_mask.astype(np.uint8)*255)
                if (wait_for_input and ic(cv2.waitKey(0)) == ord('q')):
                    wait_for_input = False
                else:
                    cv2.waitKey(1)
        
    average_precision = np.mean([score.precision for score in precision_recall_f1_scores])
    average_recall = np.mean([score.recall for score in precision_recall_f1_scores])
    average_f1 = np.mean([score.f1 for score in precision_recall_f1_scores])

    # Show all the values to get a quick overview.
    # print ("(precision, recall, f1) = ", end =" ")
    for score in precision_recall_f1_scores:
        if score.rating == Rating.GOOD:
            color_prefix, color_suffix = (TermColors.GREEN, TermColors.END)
        elif score.rating == Rating.POOR:
            color_prefix, color_suffix = (TermColors.YELLOW, TermColors.END)
        else:
            color_prefix, color_suffix = (TermColors.RED, TermColors.END)
            color_prefix += f"rc[{score.rc[0]},{score.rc[1]}]"
        # print (f"{color_prefix}({score.precision:.2f} {score.recall:.2f} {score.f1:.2f}){color_suffix}", end =" ")
    # print()

    num_good = np.count_nonzero ([score.rating == Rating.GOOD for score in precision_recall_f1_scores])
    num_samples = len(precision_recall_f1_scores)
    percentage_good = 100.0*num_good/num_samples
    # print (f"Average P,R,F1 over {num_samples} samples {average_precision:.2f}, {average_recall:.2f}, {average_f1:.2f}")
    # print (f"Percentage of great results: {percentage_good:.2f}%")
    # print (f"{percentage_good:.2f}%", end=' ', flush=True)
    return Results (percentage_good=percentage_good, average_precision=average_precision, average_recall=average_recall, average_f1=average_f1)

def main_interactive_evaluator(args, image: Path, json: Path):
    evaluator = InteractiveEvaluator()
    # labeled_image = LabeledImage (Path("generated/drawings-whitebg/img-00000-003.json"))
    if json is not None:
        labeled_image = LabeledImage (json)
        # labeled_image = LabeledImage (Path("generated/drawings/img-00000-001.json"))
        labeled_image.ensure_images_loaded()
        image_rgb = labeled_image.rendered_image
    else:
        assert (image)
        image_rgb = swap_rb(cv2.imread(str(image), cv2.IMREAD_COLOR))
        labeled_image = None
    # finder = HSVFinder(image_rgb, plot_mode=True)
    # labeled_image = None
    # image_rgb = swap_rb(cv2.imread("/home/nb/Perso/DaltonLens-Drive/Plots/Bowling.png", cv2.IMREAD_COLOR))
    finder = DeepRegressionFinder(image_rgb, args.model)
    evaluator.process_image (image_rgb, finder, labeled_image)

def main_batch_evaluation (test_dir: Path, model, output_path: Path(), save_images=False, easy_mode=False, use_baseline=False):
    test_folders = [
        test_dir / 'opencv-generated', 
        test_dir / 'mpl-generated',
        test_dir / 'mpl-generated-no-antialiasing',
        test_dir / 'mpl-generated-scatter',
        test_dir / 'opencv-generated-background',
        test_dir / 'wild',
        test_dir / 'arxiv',
    ]
    result_per_folder = {}
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=False)
    for folder in test_folders:
        print (f"{folder.name}: ", end="")
        output_folder_path = output_path / folder.name
        if save_images:
            output_folder_path.mkdir(parents=True, exist_ok=True)
        percent_good = []
        # Take the first 50 images.
        json_files = list(itertools.islice(sorted(folder.glob('*.json')), 50))
        
        # Special case for folders without labeled images.
        if len (json_files) < 1:
            if save_images:
                print (f"{folder.name}: generating images.")
                png_files = folder.glob('*.png')
                for f in png_files:
                    raw_rgb = swap_rb(cv2.imread (str(f), cv2.IMREAD_COLOR))
                    finder = DeepRegressionFinder(raw_rgb, model)
                    cv2.imwrite (str(output_folder_path / f.name), swap_rb(finder.image_rgb))
            continue
                
        for json_file in json_files:
            im = LabeledImage (Path(json_file))
            # im = LabeledImage (Path("inputs/mpl-generated/img-02778.json"))
            # im = LabeledImage (Path("generated/drawings-whitebg/img-00000-003.json"))
            im.ensure_images_loaded()
            im.compute_labels_as_rgb()
            if use_baseline:
                finder = HSVFinder(im.rendered_image, plot_mode=True)
            else:
                # FIXME: don't load the finder for every frame, that's expensive for torchscript!
                finder = DeepRegressionFinder(im.rendered_image, model)
            if save_images:
                cv2.imwrite (str(output_folder_path / json_file.with_suffix('.antialiased.png').name), swap_rb(finder.image_rgb))
            results = evaluate (im, finder, easy_mode=easy_mode)
            percent_good.append (results.percentage_good)
        result_per_folder[folder] = np.mean (percent_good)
        print (f"{folder.name}: {result_per_folder[folder]:.1f}%")
    with open(output_path / 'evaluation.txt', 'w') as f:
        f.write (" | ".join([f"{p.name}" for p in result_per_folder.keys()]))
        f.write ("\n")
        f.write (" ".join([f"{p:.1f}" for p in result_per_folder.values()]))
        f.write ("\n")
    # Dict { folder: percent_good }
    return result_per_folder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path, default=None)
    parser.add_argument('--json', type=Path, default=None)
    parser.add_argument("--batch", action='store_true')
    parser.add_argument("--model", type=str, default="regression_unetres_v4_scripted.pt")   
    parser.add_argument("--easy", action='store_true')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--baseline", action='store_true')
    args = parser.parse_args()

    global debug
    debug = args.debug
    
    # zvlog.start (('127.0.0.1', 7007))

    if args.batch:
        if args.debug:
            zvlog.start ()
        main_batch_evaluation (Path(), args.model, args.easy_mode, args.baseline)
        zvlog.waitUntilWindowsAreClosed()
    else:
        if not args.image and not args.json:
            print ("ERROR: need to specify --image or --json")
            sys.exit (1)
        if debug:
            zvlog.start ()
        main_interactive_evaluator(args, args.image, args.json)

if __name__ == "__main__":
    main ()
