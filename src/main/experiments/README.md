# Trajectory Recovery Experiments

This directory contains notebooks for experiment on trajectory recovery.

## Requirements for experiemtns using the Shanghai Telecom dataset (`sh**.ipynb`)

1. Pre-processed Datasets

If you have not downloaded the dataset already, visit [here](../../data_preprocess/shanghai/README.md#1-download-dataset) for more details.

If you wish to run the pre-process script yourself, you can run the following commands at the project root in your terminal:

```
python script.py proc --dir data/sh30-c100 --delta 30
python script.py proc --dir data/sh30-c50 --delta 30 --cluster_path exploratory_analysis/mog_50.npy
```

2. Pre-computed clusters

Pre-computed clusters available at [`exploratory_analysis/mog_100.npy`](../../../exploratory_analysis/mog_100.npy) (100 clusters) and [`exploratory_analysis/mog_50.npy`](../../../exploratory_analysis/mog_50.npy) (50 clusters). They are included in the project repository by default.

3. Pre-trained models

Pre-trained models are available at model checkpoints in [`src/ml/checkpoints`](../../ml/checkpoints/). See instruction on how to load a model [here](../../ml/experiments/README.md#pre-trained-model).

