# Trajectory Recovery

Implements trajectory recovery algorithm from [Trajectory Recovery from Ash](https://arxiv.org/abs/1702.06270). 

Since we don't know the threshold described in Section 4.3 in the paper, we skipped this approxmiation.

Euclidean distance in step-1 and step-2 is computed by converting lat-long to Cartesian coordinates using the method described [here](../ml/README.md#feature-pre-processing).

We used the `linear_sum_assignment` function in the `scipy` package to compute bipartite matching.

## Computational bottleneck at information gain computation

From our experience, the bottle neck of the algorithm is the computation of information gain.

Our current implementation for large no. of unique basestations is single-threaded and uses hash tables. We propose that it can be improved by the following ways: 1) using multiple threads; 2) vectorization using sparse matrices/tensors; 3) using the map-reduce framework for large no. of trajectories.

For small no. of unique basestations (result of clustering), our implementation do vectorization by representing the visit count of every basestations of a user using a dense vector. We perform further vectorization by batch computation. While it is possible to run this on GPU, our current implementation is fast enough for us.