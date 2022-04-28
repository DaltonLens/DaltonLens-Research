Layout:

2203/raw_figures
2203/preselection/{selected,discarded}
2203/dataset/{validated,discarded}

./extract_arxiv_figures.py data/2203/gz data/2203/raw_figures
./preselect_raw_figures.py data/2203
./dataset_from_preselection.py data/2203
