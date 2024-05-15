# GeoLife Dataset

## Methodology

### Clustering

Since the spatial points in this dataset is continuous, we apply nearest neighbor to map each point to its nearest cluster.

To compute the clusters, we use a Gaussian mixture model on at most 1000 points from each users (sampling is used for users with >1000 points).

### Missing records

Missing records of each users in each day are interpolated by linearly (expected time = Euclidean distance from a to b / average speed of the user):
- If time needed from a to b > expected time: we interpret that the user is moving at constant speed during the entire period. Where speed = Euclidean distance from a to b / time difference between 2 records
- If time needed from a to b < expected time: we interpret that the user is idle for as long as possible, than move at their average speed.

Missing records before the first known position are filled with the first known position. Missing records after the last known position are filled with the last known position.

## Script Usage

### 1. Download Dataset

Download the dataset from [Microsoft](https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/). Unzip the contents and put them in `{ROOT}/data/geolife`.

### 2. Compute clusters (Optional)

If you want to compute your own clusters, run [this notebook](../../../exploratory_analysis/geolife_bs.ipynb) to re-compute clusters with different parameters.

### 3. Run pre-process script

Run the below command to convert csv files to trajectories. One csv file will be created for each day.

```
python script.py proc --dir <output dir> --delta <discretization window> --cluster_path <cluster path>
```

The script expects `<output dir>` to be non-existant and it will create a new directory. The script will raise an error if the directory already exists. Delete or rename your folder manually if you want to replace your processed dataset files.

You can optionally provide your own pre-computed clusters. You can pass the path to your clusters via the `--cluster_path` argument. If not provided, the script will use the [pre-computed 100 clusters](../../../exploratory_analysis/geolife_mog_100.npy) by default.