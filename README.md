# DaltonLens Machine Learning

This repository collects ongoing experiments, and is by no mean production ready.

# Charts enhancement by removing anti-aliasing

The main goal is to remove line anti-aliasing so colors become flat and can be easily segmented in DaltonLens.

## Setup with conda

```
conda config --set auto_activate_base false
conda config --set channel_priority strict
conda install mamba

conda create --name dl python=3.7
conda config --add channels pytorch
mamba install pytorch torchvision torchaudio cudatoolkit matplotlib tensorboard jupyterlab ipykernel tqdm

# Register the conda env for jupyter notebooks
python -m ipykernel install --user --name=dl

conda activate dl
```

Then install the packages in edit mode:

```
pip install -e charts
```

It may require DaltonLens-Python to be installed.

## Training a model

Edit and run `charts/next_train.py`.

There were experiments with fast.ai and pytorch-lightning (in the attic), but the latest only uses vanilla pytorch.

Logs can be visualized with `tensorboard --logdir=charts/logs`.

Evolution of the images across epochs can be visualized with e.g. `zv charts/logs/xxx/**/*.png`.

## Training data

Training data is expected under `charts/inputs/train`. 

- `charts/inputs/generate/`: scripts to generate data with OpenCV drawings and matplotlib are under .

- `charts/inputs/arxiv/`: scripts to collect data from arxiv figures. See [charts/inputs/arxiv/README.md](charts/inputs/arxiv/README.md).

A backup of the last generated data is available [on google drive](https://drive.google.com/drive/folders/1zOzXQJAgX6LpNZugkMCYk8EaYc70I27k?usp=sharing).

## Testing a model

**TODO**: these scripts needs to be adapted to the latest Gated/Masked version.

`scripts/export_to_torchscript.py` generates an onnx file from a checkpoint.

`scripts/test_onnx.py` loads pretrained models with various backends, like cv2.dnn and onnxruntime .
