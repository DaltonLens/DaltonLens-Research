Layout:

2203/raw_figures
2203/preselection/{selected,discarded}
2203/dataset/{validated,discarded}

Data used so far:
- arXiv_src_2203_{001,002,003,004}.tar to generate training data.
- arXiv_src_2203_{005,006,007} to generate test data.

They were downloaded from the arxiv s3 buckets:
https://arxiv.org/help/bulk_data_s3

s3cmd --requester-pays ls s3://arxiv/src/
s3cmd --requester-pays get --skip-existing s3://arxiv/src/arXiv_src_2203_'*'.tar

cd ../rawdata/arxiv/2203/gz && tar xvf arXiv_src_2203_001.tar

./extract_arxiv_figures.py ../rawdata/arxiv/2203/gz data/2203/raw_figures
./preselect_raw_figures.py ../rawdata/arxiv/data/2203
./dataset_from_preselection.py ../rawdata/arxiv/data/2203
