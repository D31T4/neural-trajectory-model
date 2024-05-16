# P14: Trajectory Recovery from Ash: In the Age of AI

## Introduction

This repository implements the trajectory recovery algorithm described in [Trajectory Recovery From Ash: User Privacy Is NOT Preserved in Aggregated Mobility Data](https://arxiv.org/abs/1702.06270) as our baseline. In our approach, we model human trajectories using a architecture similar to [CLIP](https://arxiv.org/abs/2103.00020) and we use the negative log-likelihood predicted by the model as the cost matrix proposed by the baseline framework.

Our work simulates a scenario where an attacker has access to a trajectory dataset with similar distribution to our target aggregated location data. It demonstrates the potential usage of neural networks in human trajectory recovery from aggregated datasets, and achieves better performance then the baseline.

## Dependencies

The Python version used in this project is 3.9.0.

See [`requirements.txt`](requirements.txt) for external dependencies.

## Disclaimer

This project is used to fulfill the requirements of COMP3900/9900 capstone project at the University of New South Wales.