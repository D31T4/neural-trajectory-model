# Trajectory Recovery Experiments

This directory contains notebooks for experiment on trajectory recovery.

## Requirements

### Requirements for experiments using the Shanghai Telecom dataset (`sh**.ipynb`)

1. Pre-processed Datasets

If you have not downloaded the dataset already, visit [here](../../data_preprocess/shanghai/README.md#1-download-dataset) for more details.

If you wish to run the pre-process script yourself, you can run the following commands at the project root in your terminal:

```
python script.py proc --dir data/sh30-c100 --delta 30
python script.py proc --dir data/sh30-c50 --delta 30 --cluster_path exploratory_analysis/mog_50.npy
python script.py proc --dir data/sh30-cinf --detla 30 --cluster_path false
```

2. Pre-computed clusters

Pre-computed clusters available at [`exploratory_analysis/mog_100.npy`](../../../exploratory_analysis/mog_100.npy) (100 clusters) and [`exploratory_analysis/mog_50.npy`](../../../exploratory_analysis/mog_50.npy) (50 clusters). They are included in the project repository by default.

3. Pre-trained models

Pre-trained models are available at model checkpoints in [`src/ml/checkpoints`](../../ml/checkpoints/). See instruction on how to load a model [here](../../ml/experiments/README.md#pre-trained-model).


### Requirements for experiments using the GeoLife dataset

1. Pre-processed Datasets

If you wish to run the pre-process script yourself, you can run the following commands at the project root in your terminal:

```
python script.py proc --dir data/geolife-c100 --delta 30
```

2. Pre-computed clusters

Pre-computed clusters available at [`exploratory_analysis/geolife_mog_100.npy`](../../../exploratory_analysis/geolife_mog_100.npy) (100 clusters). They are included in the project repository by default.

## Data used for testing

In all datasets, we only consider "persistent" users (users who have records everyday during the testing period).

### Shanghai Telecom dataet

Trajectories from 2014-06-18 to 2014-07-02 (1309 trajectories in total) are used for testing.

### GeoLife dataset

#### `geolife-c100.ipynb`:

Trajectories from 2009-02-13 to 2009-02-19 (48 trajectories in total) are used for testing.

#### `geolife-c100-merged.ipynb`:

Due to the small amount of users in the dataset, we use the set of all trajectories from Thu to Wed are used for testing (4759 trajectories in total). This means multiple trajectories from the same users are included in the test set.