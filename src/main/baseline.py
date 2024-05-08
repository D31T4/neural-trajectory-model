from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import tqdm
import math

from src.data_preprocess.trajectory import Trajectory, from_numpy
from src.data_preprocess.point import CoordinateMap, LatLong

from typing import Callable

AggregatedObservation = list[tuple[LatLong, int]]
GetCostMatrix = Callable[[np.ndarray, np.ndarray, int], np.ndarray]

class RecoveryFromAshDistance:
    '''
    Implement cost matrix in step-1 and step-2 in Trajectory Recovery from Ash
    '''

    def __init__(self, ref: LatLong, day_time: int):
        '''
        Args:
        ---
        - ref: reference point for cartesian coordinate conversion
        - day_time: earliest timestamp to be considered day
        '''
        self.coord_map = CoordinateMap(ref)
        self.day_time = day_time

    def night_time_distance(self, trajectories: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        '''
        Implements night time cost matrix in Trajectory Recovery from Ash

        Args:
        ---
        - trajectories: [n1, L, (lat, long)]
        - candidates: [n2, (lat, long)]

        Returns:
        ---
        - distance: [n1, n2]
        '''
        candidates = self.coord_map.to_cartesian(candidates)
        next_loc = self.coord_map.to_cartesian(trajectories[:, -1])

        return euclidean_distances(next_loc, candidates)

    def day_time_distance(self, trajectories: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        '''
        Implements day time cost matrix in Trajectory Recovery from Ash

        Args:
        ---
        - trajectories: [n1, L, (lat, long)]
        - candidates: [n2, (lat, long)]

        Returns:
        ---
        - distance: [n1, n2]
        '''
        candidates = self.coord_map.to_cartesian(candidates)
        
        prev_loc = self.coord_map.to_cartesian(trajectories[:, -2])
        current_loc = self.coord_map.to_cartesian(trajectories[:, -1])

        vel = current_loc - prev_loc
        next_loc = current_loc + vel

        return euclidean_distances(next_loc, candidates)

    def compute(self, trajectories: np.ndarray, candidates: np.ndarray, tstamp: int) -> np.ndarray:
        '''
        compute cost matrix

        Args:
        ---
        - trajectories: [n1, L, (lat, long)]
        - candidates: [n2, (lat, long)]
        - tstamp: timestamp

        Returns:
        ---
        - cost matrix [n1, n2]
        '''
        if tstamp < self.day_time:
            return self.night_time_distance(trajectories, candidates)
        else:
            return self.day_time_distance(trajectories, candidates)


def get_aggregated_observations(trajectories: list[Trajectory], tstamp: int) -> AggregatedObservation:
    '''
    get aggregated observations from ground truth

    Args:
    ---
    - trajectories: list of trajectories
    - tstamp: time stamp

    Returns:
    ---
    - aggregated observations
    '''
    out: dict[LatLong, int] = dict()

    for trajectory in trajectories:
        point = trajectory[tstamp]
        out[point] = out.get(point, 0) + 1

    return [*out.items()]

class DailyTrajectoryRecovery:
    '''
    single day trajectory recovery.
    '''
    def __init__(
        self, 
        get_observations: Callable[[int], AggregatedObservation], 
        cost_matrix: GetCostMatrix, 
        sequence_length: int, 
        sequence_count: int
    ):
        '''
        Args:
        ---
        - get_observations: function to get aggregated observations `(timestamp: int) => aggregated observations: list[(basestation lat-long, user_count)]`
        - cost_matrix: function to obtain cost matrix. `(trajectories[n1], basestations[n2], time stamp of prediction) => cost_matrix[n1, n2]`
        - sequence_length: trajectory length
        - sequence_count: no. of trajectory
        '''
        self.result = np.zeros((sequence_count, sequence_length, 2))
        self.get_observations = get_observations
        self.cost_matrix = cost_matrix
        
        self.sequence_length = sequence_length
        self.sequence_count = sequence_count

    @staticmethod
    def unaggregate_observations(observations: AggregatedObservation) -> tuple[np.ndarray, np.ndarray]:
        '''
        unagregate observations

        Args:
        ---
        - aggregated_observations: list of basestations and no. of users

        Returns:
        ---
        - list of basestations
        - reference indices
        '''
        out_len = sum(e[1] for e in observations)
        out = np.zeros((out_len, 2))
        indices = np.zeros(out_len, dtype=int)
        j = 0

        for i, (loc, count) in enumerate(observations):
            assert count > 0
            
            out[j:j + count, 0] = loc[0]
            out[j:j + count, 1] = loc[1]
            indices[j:j + count] = i

            j += count

        return out, indices


    def run(self, verbose: bool = True):
        '''
        Run algorithm

        Args:
        ---
        - verbose: show progress on tqdm
        '''
        self.result[:, 0], _ = DailyTrajectoryRecovery.unaggregate_observations(self.get_observations(0))

        for tstamp in tqdm.trange(1, self.sequence_length, desc='per day recovery', disable=not verbose):
            observations = self.get_observations(tstamp)
            unique_observations = np.array([c[0] for c in observations])
            
            cost_matrix = self.cost_matrix(self.result[:, :tstamp], unique_observations, tstamp)

            observations, indices = DailyTrajectoryRecovery.unaggregate_observations(observations)
            cost_matrix = cost_matrix[:, indices]

            row_idx, col_idx = linear_sum_assignment(cost_matrix, maximize=False)
            row_idx: np.ndarray = row_idx # type hint hack
            col_idx: np.ndarray = col_idx

            # append point to trajectory
            self.result[row_idx, tstamp] = observations[col_idx, :]

        return self

    def get_predicted_trajectories(self) -> list[Trajectory]:
        '''
        get list of predicted trajectories
        '''
        return from_numpy(self.result)

class CrossDayTrajectoryRecovery:
    '''
    Implements cross-day trajectory recovery from Trajectory Recovery from Ash. Use for large vocab size.

    use hash tables in entropy computation for large no. of base stations. 
    ~1 hr to run on shanghai 100 clusters test set (1.3k trajectories)
    '''
    def __init__(self, trajectories: list[list[Trajectory]]):
        '''
        Args:
        ---
        - trajectories: days of trajectories
        '''
        self.result = None
        self.trajectories = trajectories

        trajectory_count = len(trajectories[0])
        assert all(len(t) == trajectory_count for t in trajectories)

    @staticmethod
    def entropy(trajectory: np.ndarray) -> float:
        '''
        Implements trajectory entropy in Trajectory Recovery from Ash

        Args:
        ---
        - trajectory

        Returns:
        ---
        - entropy
        '''
        _, counts = np.unique(trajectory, axis=0, return_counts=True)
        counts = counts / trajectory.shape[0]
        return -(np.log(counts) * counts).sum()

    @staticmethod
    def recovery_from_ash_information_gain(l0: list[Trajectory], l1: list[Trajectory]) -> np.ndarray:
        '''
        Implements information gain in step-3 of Trajectory Recovery from Ash

        Note: seem to be the bottle neck of the algorithm. maybe optimize it using sparse matrix?

        Args:
        ---
        - l0: trajectories[n1]
        - l1: trajectories[n2]

        Returns:
        ---
        - information gain [n1, n2]
        '''
        ig = np.zeros((len(l0), len(l1)))

        entropy_cache0 = np.zeros(len(l0))
        entropy_cache1 = np.zeros(len(l1))

        l0 = np.array(l0)
        l1 = np.array(l1)

        for i in range(l0.shape[0]):
            entropy_cache0[i] = CrossDayTrajectoryRecovery.entropy(l0[i])

        for i in range(l1.shape[0]):
            entropy_cache1[i] = CrossDayTrajectoryRecovery.entropy(l1[i])

        for i in range(l0.shape[0]):
            for j in range(l1.shape[0]):
                ig[i, j] = CrossDayTrajectoryRecovery.entropy(np.concatenate((l0[i], l1[j]), axis=0)) - (entropy_cache0[i] + entropy_cache1[j]) / 2

        return ig

    def run(self, verbose: bool = True):
        '''
        run algorithm. Get result in `result` property.
        '''
        self.result = [t.copy() for t in self.trajectories[0]]

        for i in tqdm.trange(1, len(self.trajectories), desc='cross-day recovery', disable=not verbose):
            cost_matrix = CrossDayTrajectoryRecovery.recovery_from_ash_information_gain(self.result, self.trajectories[i])

            row_idx, col_idx = linear_sum_assignment(cost_matrix, maximize=False)
            row_idx: np.ndarray = row_idx # type hint hack
            col_idx: np.ndarray = col_idx

            # concat matched trajectories
            for row, col in zip(row_idx, col_idx):
                self.result[row].points += self.trajectories[i][col].points.copy()

class DenseCrossDayTrajectoryRecovery(CrossDayTrajectoryRecovery):
    '''
    vectorize entropy computation using dense arrays

    better perf for small vocal size.
    ~5 min to run on shanghai 100 clusters test set (1.3k trajectories)
    ~1 min with chunk_size = 100

    Memory usage:
    - O(d x chunk_size ^ 2) memory when computing information gain.
    - O(N x d) for location counts.

    Where N = no. of trajectories, d = vocab size
    '''
    def __init__(self, trajectories: list[list[Trajectory]], map: dict[LatLong, int], chunk_size: int = 1):
        '''
        Args:
        ---
        - trajectories
        - map: hash table lat-long => index
        - chunk_size: chunk size for entropy computation. set > 1 for chunked computation
        '''
        super().__init__(trajectories)
        
        self.map = map
        self.chunk_size = chunk_size

    @staticmethod
    def entropy(counts: np.ndarray):
        '''
        vectorized entropy computation

        Args:
        ---
        - counts: array of counts

        Returns:
        ---
        - entropy
        '''
        counts = counts / counts.sum(axis=-1)[..., None]
        return -np.nansum(np.log(counts) * counts, axis=-1)
    
    def recovery_from_ash_information_gain(self, l1: list[Trajectory]):
        '''
        Implements information gain in step-3 of Trajectory Recovery from Ash

        Args:
        ---
        - l1: trajectories[n2]

        Returns:
        ---
        - information gain [n1, n2]
        '''
        l1 = np.array(l1)

        next_counts = self.get_counts(l1)

        entropy_cache0 = DenseCrossDayTrajectoryRecovery.entropy(self.counts)
        entropy_cache1 = DenseCrossDayTrajectoryRecovery.entropy(next_counts)

        ig = np.zeros((len(self.counts), len(l1)))

        for i in range(self.counts.shape[0]):
            for j in range(l1.shape[0]):
                ig[i, j] = DenseCrossDayTrajectoryRecovery.entropy(self.counts[i] + next_counts[j]) - (entropy_cache0[i] + entropy_cache1[j]) / 2

        return ig, next_counts
    
    def recovery_from_ash_information_gain_chunked(self, l1: list[Trajectory]):
        '''
        Implements information gain in step-3 of Trajectory Recovery from Ash with chunked vectorization.

        Note: seem to be the bottle neck of the algorithm. maybe optimize it using sparse matrix?

        Args:
        ---
        - l1: trajectories[n2]

        Returns:
        ---
        - information gain [n1, n2]
        '''
        l1 = np.array(l1)

        next_counts = self.get_counts(l1)

        entropy_cache0 = DenseCrossDayTrajectoryRecovery.entropy(self.counts)
        entropy_cache1 = DenseCrossDayTrajectoryRecovery.entropy(next_counts)

        chunk_size = min(self.chunk_size, self.counts.shape[0])

        ig = np.zeros((len(self.counts), len(l1)))

        # chunked computation
        for i in range(math.ceil(self.counts.shape[0] / chunk_size)):
            i = i * chunk_size

            for j in range(math.ceil(next_counts.shape[0] / chunk_size)):
                j = j * chunk_size

                # vectorized entropy computation
                pairwise_merged_entropy = DenseCrossDayTrajectoryRecovery.entropy(
                    self.counts[i:i + chunk_size, None] + next_counts[None, j:j + chunk_size]
                )

                ig[i:i + chunk_size, j:j + chunk_size] = pairwise_merged_entropy - (entropy_cache0[i:i + chunk_size, None] + entropy_cache1[None, j:j + chunk_size]) / 2

        return ig, next_counts

    def get_counts(self, trajectories: np.ndarray):
        '''
        get vocab count

        Args:
        ---
        - trajecctories: [N x L x 2]

        Returns:
        ---
        - counts: [N x d]
        '''
        counts = np.zeros((trajectories.shape[0], len(self.map)))

        for i in range(len(self.result)):
            locs, cnts = np.unique(trajectories[i], axis=0, return_counts=True)

            for loc, count in zip(locs, cnts):
                loc_idx = self.map[tuple(loc.tolist())]
                counts[i, loc_idx] += count

        return counts

    def run(self, verbose: bool = True):
        self.result = [t.copy() for t in self.trajectories[0]]

        self.counts = self.get_counts(np.array(self.trajectories[0]))

        if self.chunk_size == 1:
            information_gain = self.recovery_from_ash_information_gain
        else:
            information_gain = self.recovery_from_ash_information_gain_chunked

        for i in tqdm.trange(1, len(self.trajectories), desc='cross-day recovery', disable=not verbose):
            cost_matrix, next_counts = information_gain(self.trajectories[i])

            row_idx, col_idx = linear_sum_assignment(cost_matrix, maximize=False)
            row_idx: np.ndarray = row_idx # type hint hack
            col_idx: np.ndarray = col_idx

            # concat matched trajectories
            for row, col in zip(row_idx, col_idx):
                self.result[row].points += self.trajectories[i][col].points.copy()

            self.counts[row] += next_counts[col]

        self.counts = None

