from pathlib import Path

def regression_datasets (root_dir: Path, name: str):
    
    datasets = dict(
        v4=[
            root_dir / 'inputs' / 'train-v4' / 'opencv-generated',
            root_dir / 'inputs' / 'train-v4' / 'opencv-generated-background',
            root_dir / 'inputs' / 'train-v4' / 'mpl-generated',
            root_dir / 'inputs' / 'train-v4' / 'mpl-generated-no-antialiasing',
            root_dir / 'inputs' / 'train-v4' / 'mpl-generated-scatter',
        ],

        v5_no_arXiv=[
            root_dir / 'inputs' / 'train-v5' / 'opencv-drawings/320x240',
            root_dir / 'inputs' / 'train-v5' / 'opencv-drawings/640x480',
            root_dir / 'inputs' / 'train-v5' / 'opencv-drawings/640x480_upsampled',
            root_dir / 'inputs' / 'train-v5' / 'mpl-no-aa/320x240',
            root_dir / 'inputs' / 'train-v5' / 'mpl-no-aa/640x480',
            root_dir / 'inputs' / 'train-v5' / 'mpl/320x240',
            root_dir / 'inputs' / 'train-v5' / 'mpl/640x480',
            root_dir / 'inputs' / 'train-v5' / 'mpl/640x480_upsampled',
            root_dir / 'inputs' / 'train-v5' / 'opencv-drawings-bg/320x240',
            root_dir / 'inputs' / 'train-v5' / 'opencv-drawings-bg/640x480',
            root_dir / 'inputs' / 'train-v5' / 'mpl-scatter/320x240',
            root_dir / 'inputs' / 'train-v5' / 'mpl-scatter/640x480',
            root_dir / 'inputs' / 'train-v5' / 'mpl-scatter/640x480_upsampled',
        ],

        v5=[
            root_dir / 'inputs' / 'train-v5' / 'arxiv/320x240',
            root_dir / 'inputs' / 'train-v5' / 'arxiv/320x240_bg',
            root_dir / 'inputs' / 'train-v5' / 'arxiv/640x480_bg',
            root_dir / 'inputs' / 'train-v5' / 'arxiv/640x480',
            root_dir / 'inputs' / 'train-v5' / 'arxiv/640x480_bg_upsampled',
            root_dir / 'inputs' / 'train-v5' / 'arxiv/640x480_upsampled',
            root_dir / 'inputs' / 'train-v5' / 'opencv-drawings/320x240',
            root_dir / 'inputs' / 'train-v5' / 'opencv-drawings/640x480',
            root_dir / 'inputs' / 'train-v5' / 'opencv-drawings/640x480_upsampled',
            root_dir / 'inputs' / 'train-v5' / 'mpl-no-aa/320x240',
            root_dir / 'inputs' / 'train-v5' / 'mpl-no-aa/640x480',
            root_dir / 'inputs' / 'train-v5' / 'mpl/320x240',
            root_dir / 'inputs' / 'train-v5' / 'mpl/640x480',
            root_dir / 'inputs' / 'train-v5' / 'mpl/640x480_upsampled',
            root_dir / 'inputs' / 'train-v5' / 'opencv-drawings-bg/320x240',
            root_dir / 'inputs' / 'train-v5' / 'opencv-drawings-bg/640x480',
            root_dir / 'inputs' / 'train-v5' / 'mpl-scatter/320x240',
            root_dir / 'inputs' / 'train-v5' / 'mpl-scatter/640x480',
            root_dir / 'inputs' / 'train-v5' / 'mpl-scatter/640x480_upsampled',
        ],
    )

    return datasets[name]
