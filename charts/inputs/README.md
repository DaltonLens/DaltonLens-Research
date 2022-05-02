# Generate training samples

## Ground truth format

- 1 json file with the color of each label

- 1 label image (grayscale), one label per pixel. 0 is the "background". The background is anything that is not rendered from a vector path. The frontier is tricky when large uniform areas are rendered in the plot, but they'll be considered foreground. The foreground is expected to be generated with a stroke width of at least 2. So lines that are thinner than that in the antialiased image need to be expanded.

- 1 antialiased rendering (color PNG)

- Optional: 1 aliased rendering. Can always be re-generated from the label image and the anti-aliased image (required for non-uniform backgrounds). See `generate_aliased_from_labels.py`.

## From OpenCV drawings

`generate_opencv.py` . Generates drawings (lines, rectangles, circles) with
varying stroke width and locations.

[imagenette](https://github.com/fastai/imagenette) has been used to generate the version with an image background.

The code is a bit complex for historical reason, these days it'd be enough to
just render one with cv::LINE_AA and once without.


## From matplotlib

`generate_matplotlib.py`

Here also the code is a bit more complex than strictly necessary to really
create one label per layer. The rendering is always anti-aliased, but we
classify any pixel that changed from the background color as part of the
foregroup path.

## From arxiv

See the [README.md](arxiv/README.md) there.

## Useful scripts

Convert from old ground truth format to the new one:

`./convert_from_labels.sh ../train-old-filledmask/opencv-generated-background opencv-generated-background`
