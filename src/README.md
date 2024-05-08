# P14：Trajectory Recovery from Ash: In the Age of AI

## Database Download Links

[Telecom Shanghai Dataset](https://www.kaggle.com/datasets/mexwell/telecom-shanghai-dataset) 

Contains more than 7.2 million records of accessing the Interent through 3,233 base stations from 9,481 mobile phones for six months. 

[GeoLife GPS Trajectories](https://www.microsoft.com/en-us/download/details.aspx?id=52367)

This GPS trajectory dataset was collected in (Microsoft Research Asia) Geolife project by 182 users in a period of over three years (from April 2007 to August 2012).

## Dataset Preprocessing
1. **Shanghai**: See [here](data_preprocess/shanghai/README.md#script-usage).

2. **Geolife**: *data_preprocess/preprocessing_Geolife.ipynb*
    - Input: Downloaded data in current directory
    - Preprocesssing steps: Data reading, outliers removal, data interpolation, data discretization, data export
    - Output: CSV file for each day

## Baseline model
1. **Algorithm implementation**: See [`src/main`](main/README.md)

2. **Evaluation**: See [`src/main/experiments`](main/experiments/README.md)

## LSTM model
*ml/lstm.ipynb*
- Prerequisite: Processed data in csv file with standardized format, required library
- Functionality: Include all functions and procedures for data input pipeline, model definition, model training, model saving and model predictions

*ml/visualize_lstm.ipynb*
- Prerequisite: Predicted dataframe and ground truth dataframe in standardized format, required library
- Functionality: Display prediction results for 1 step or whole day

## Transformer based model

See [here](ml/experiments/README.md).