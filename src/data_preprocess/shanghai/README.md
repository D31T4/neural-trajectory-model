# Shanghai Telecom Dataset

## Methodology

### Clustering

The no. of concurrent users in each base stations are too low to evaluate the reconstruction attack as most users can be uniquely identified by their connected basestations. 

Hence, we apply clustering to aggregate nearby base stations. We used Gaussian mixture models on the all base station locations to compute the clusters. To compute high quality clusters, we applied the Gaussian mixture algorithm in the scikit-learn package with spherical variance on the set of unique base stations. The Gaussian mixture algorithm consists of 2 steps: 1) compute k-means clustering to obtain the initial clusters; 2) iterative expectation maximization. The distribution of resultant clusters visually resembles the original base stations in the dataset, where the density is higher in urban areas.

And finally, we compute the nearest neighbor of each base station to find its assigned cluster. We use nearest neighbor due to 2 reasons: 
1. avoid ambiguity of overlapping clusters.
2. sub-linear time complexity using pre-computed index.

To choose a good number of clusters, we plot the effects of the number of clusters on the number of non-stationary persistent users (users who have been to at least 3 locations every day in a chosen week), and the average number of concurrent users. We observed that using a small number of clusters would increase the probability of base stations being assigned to the same base stations, resulting in users appearing to be stationary as the base stations in their trajectories being assigned in the same cluster. Since stationary users are not desirable for assessing the day time recovery part of the algorithm, we ended up using 100 clusters which we believe is a good balance between reducing identifying power of individual base stations, and preserving movement of users across base stations.

### Missing records

Missing records of each users in each day are interpolated by filling with the last known position. We used this method instead of linear interpolation in Trajectory Recovery from Ash because we think this dataset is too sparse for the usage of linear interpolation.

Missing records before the first known position are filled with the first known position.

## Script Usage

### 1. Download dataset

Download dataset from [Google drive](https://drive.google.com/file/d/1TWD3QDBrsn90zxbDom94BF4fR-NOp0Pi) provided by dataset author. The URL can be found [here](http://sguangwang.com/TelecomDataset.html).

### 2. Convert `.xlsx` files to `.csv` files

Run the below command to convert excel files to csv files for faster read.

```
python script.py csv --dir <dir to excel files>
```

### 3. Compute clusters (Optional)

If you want to compute your own clusters, run [this notebook](../../../exploratory_analysis/shanghai_bs_anal.ipynb) to re-compute clusters with different parameters.

### 4. Run pre-process script

Run the below command to convert csv files to trajectories. One csv file will be created for each day.

```
python script.py proc --dir <output dir> --delta <discretization window> --cluster_path <optional cluster path>
```

The script expects `<output dir>` to be non-existant and it will create a new directory. The script will raise an error if the directory already exists. Delete or rename your folder manually if you want to replace your processed dataset files.

You can optionally provide your own pre-computed clusters. You can pass the path to your clusters via the `--cluster_path` argument. If not provided, the script will use the [pre-computed 100 clusters](../../../exploratory_analysis/mog_100.npy) by default. Input `false` if you do not want to perform spatial aggregation.

## Download

Pre-processed data is available on [my Google drive](https://drive.google.com/drive/folders/1jSsBABpP-GBpztqmaZp_mXET694bIeHu?usp=sharing). May get deleted if I ran out of space.

Data pre-processed with { 50 clusters, 100 clusters }, and temporal resolution of { 30 min, 90 min, 120 min } are available for download.
