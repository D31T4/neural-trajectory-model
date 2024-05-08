# Deep Learning Experiments

This directory contains notebooks used for deep learning experiments.

## Requirements

1. Pre-processed Datasets

If you have not downloaded the dataset already, visit [here](../../data_preprocess/shanghai/README.md#1-download-dataset) for more details.

If you wish to run the pre-process script yourself, you can run the following commands at the project root in your terminal:

```
python script.py proc --dir data/sh30-c100 --delta 30
python script.py proc --dir data/sh30-c50 --delta 30 --cluster_path exploratory_analysis/mog_50.npy
```

2. Pre-computed clusters

Pre-computed clusters available at [`exploratory_analysis/mog_100.npy`](../../../exploratory_analysis/mog_100.npy) (100 clusters) and [`exploratory_analysis/mog_50.npy`](../../../exploratory_analysis/mog_50.npy) (50 clusters). They are included in the project repository by default.

## Training Configuration

See `config.py` and `checkpoint.py` for training related settings.

## Pre-trained model

A pre-trained model with embedding dimension of 128 are available from the checkpoint at [`checkpoints/sh30-c100_best.pt`](../checkpoints/sh30-c100_best.pt).

The model is trained with the Shanghai Telecom Dataset pre-processed with 100 clusters and temporal resolution of 30 min.

You can load the model parameters using the snippet below.

```{python}
model_dim = 128

model = TrajectoryModel(
    base_station_embedding=BaseStationEmbedding(
        feat_dim=(2, 64),
        context_dim=(31, 48),
        out_dim=model_dim,
        layer_norm=True
    ),
    trajectory_encoder=TransformerTrajectoryEncoder(
        in_dim=model_dim,
        max_len=SEQ_LENGTH,
        hid_dim=(model_dim, model_dim * 2, 8),
        do_prob=0.2,
        n_blocks=4
    ),
)

best_state = torch.load(<your path to the checkpoint>)

model.load_state_dict(best_state['model'])
```

## Notes

Experiments on different model dimension is done by making ad-hoc changes directly on the experiment notebooks. Therefore not provided.