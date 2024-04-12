"""
Discovering Routines of Fixed Length.

This script allows to discover routines of fixed length in a time series. The algorithm is based on the paper "An incremental algorithm for discovering routine behaviors from smart meter data" by Jin Wang, Rachel Cardell-Oliver and Wei Liu.

The algorithm is based on the following steps:

    * Extract subsequences of fixed length from the time series.
    * Group the subsequences into clusters based on their magnitude and maximum absolute distance.
    * Filter the clusters based on their frequency.
    * Test and handle overlapping clusters.

The algorithm is implemented in the class DRFL, which has the following methods and parameters:

The parameters:
    * _m: Length of each secuence
    * _R: distance threshold
    * _C: Frequency threshold
    * _G: magnitude threshold
    * _epsilon: Overlap Parameter

Public methods:
    * fit: Fit the time series to the algorithm.
         Parameters:
            - time_series: Temporal data.
    * show_results: Show the results of the algorithm.
    * get_results: Returns the object Routines, with the discovered routines.
    * plot_results: Plot the results of the algorithm.
        Parameters:
            - title_fontsize: `Optional[int]`. Size of the title plot.
            - ticks_fontsize: `Optional[int]`. Size of the ticks.
            - labels_fontsize: `Optional[int]`. Size of the labels.
            - figsize: `Optional[tuple[int, int]]`. Size of the figure.
            - xlim: `Optional[tuple[int, int]]`. Limit of the x axis with starting points.
            - save_dir: `Optional[str]`. Directory to save the plot.

"""
import sys
import datetime
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from typing import Union, Optional
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import gridspec
import warnings

from itertools import product
from src.structures import Subsequence, Sequence, Cluster, Routines, HierarchyRoutine

sys.path.append("../")


class DRFL:
    """
    Discovering Routines of Fixed Length.

    This class allows to discover routines of fixed length in a time series. The algorithm is based on the paper "An incremental algorithm for discovering routine behaviors from smart meter data" by Jin Wang, Rachel Cardell-Oliver and Wei Liu.

    The algorithm is based on the following steps:

            * Extract subsequences of fixed length from the time series.
            * Group the subsequences into clusters based on their magnitude and maximum absolute distance.
            * Filter the clusters based on their frequency.
            * Test and handle overlapping clusters.

    Parameters:
        * m: Length of each secuence
        * R: distance threshold
        * C: Frequency threshold
        * G: magnitude threshold
        * epsilon: Overlap Parameter

    Methods
    _______
        * fit: Fit the time series to the algorithm.
        * show_results: Show the results of the algorithm.
        * get_results: Returns the object Routines, with the discovered routines.
        * plot_results: Plot the results of the algorithm.
        * estimate_distance: Estimate the customized distance from the obtained centroids and the target centroids.

    Examples:
    --------

        >>> import pandas as pd

        >>> time_series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> time_series.index = pd.date_range(start="2024-01-01", periods=len(time_series))
        >>> drfl = DRFL(m=3, R=2, C=3, G=4, epsilon=0.5)
        >>> drfl.fit(time_series)
        >>> drfl.show_results()
        >>> drfl.plot_results()

    """

    def __init__(self, m: int, R: Union[float, int], C: int, G: Union[float, int], epsilon: float,
                 L: Union[float, int] = 0):
        """
        Initialize the DRFL algorithm.

        Parameters:
            * m: `int`. Length of each subsequence.
            * R: `float` or `int`. distance threshold.
            * C: `int`. Frequency threshold.
            * G: `float` or `int`. magnitude threshold.
            * epsilon: `float`. Overlap parameter.
            * L: `float` or `int`. inverse magnitude threshold (default is 0).

        Examples:
            >>> drfl = DRFL(m=3, R=2, C=3, G=4, epsilon=0.5)
        """

        # Check the validity of the parameters
        params = locals()
        self._check_params(**params)

        self._m: int = m
        self._R: int | float = R
        self._C: int = C
        self._G: int | float = G
        self._epsilon: float = epsilon

        self._L: int | float = L
        self.__routines: Routines = Routines()
        self.__sequence: Sequence = Sequence()

        self.__already_fitted: bool = False
        self.time_series: pd.Series = None

    @staticmethod
    def _check_params(**kwargs):
        """
        Check the validity of the parameters for the DRGS algorithm.

        Parameters:
            * kwargs: `dict`. Dictionary containing the parameters for the DRGS algorithm.

        Raises:
            TypeError: If any parameter is not of the correct type.
            ValueError: If any parameter has an invalid value.
        """

        # Check the validity of the parameters
        for key, value in kwargs.items():
            # Check if length range is a tuple of two integers
            if key == "length_range":
                if not isinstance(value, tuple):
                    raise TypeError("length_range must be a tuple")

                if len(value) != 2:
                    raise ValueError("length_range must be a tuple with two values")

                if not all(isinstance(v, int) for v in value):
                    raise TypeError("length_range values must be integers")

                # Check if the values are valid
                if value[0] < 2 or value[1] < value[0]:
                    raise ValueError("Invalid length_range values")

            # Check if the distance threshold, magnitude and inverse magnitude are integer or a float
            if key in ["R", "G", "L"]:
                if not isinstance(value, (int, float)):
                    raise TypeError(f"{key} must be an integer or a float")

                if value < 0:
                    raise ValueError(f"{key} must be greater or equal than 0")

            # Check if the frequency threshold is an integer greater than 1
            if key == "C":
                if not isinstance(value, int):
                    raise TypeError("C must be an integer")

                if value < 1:
                    raise ValueError("C must be greater or equal than 1")

            # Check if the epsilon is a float between 0 and 1
            if key == "epsilon":
                if not isinstance(value, (int, float)):
                    raise TypeError("epsilon must be a float")

                if value < 0 or value > 1:
                    raise ValueError("epsilon must be between 0 and 1")

    @staticmethod
    def _check_type_time_series(time_series: pd.Series) -> None:
        """
        Check the type of the time series.

        Parameters:
            * time_series: `pd.Series`. Temporal data.

        Raises:
            TypeError: If the time series is not a `pandas Series` with `DatetimeIndex` at the index.

        Notes:
            This method is private and cannot be accessed from outside the class.

        Examples:
            >>> import pandas as pd
            >>> time_series = pd.Series([1, 2, 3, 4, 5])
            >>> DRFL._check_type_time_series(time_series)
            TypeError: time_series must be a pandas Series with a DatetimeIndex
        """

        if not isinstance(time_series, pd.Series):
            raise TypeError("time_series must be a pandas Series")

        if not isinstance(time_series.index, pd.DatetimeIndex):
            raise TypeError("time_series index must be a pandas DatetimeIndex")

    @staticmethod
    def _check_plot_params(**kwargs: dict) -> None:
        """
        Check the validity of the parameters for the plot_results method.

        Parameters:
            * kwargs: `dict`. The parameters to check.

        Raises:
            TypeError: If some parameter is not valid or has an invalid value.

        """
        # Get the arguments of the method and check their validity
        saved_args = locals()
        integer_params = ["title_fontsize", "xticks_fontsize", "yticks_fontsize", "labels_fontsize", "coloured_text_fontsize", "text_fontsize"]
        tuple_params = ["figsize", "xlim"]

        # Check the validity of the parameters
        for key, value in saved_args["kwargs"].items():
            # Check if the numerical parameters are integers
            if key in integer_params:
                if not isinstance(value, int):
                    raise TypeError(f"{key} must be an integer. Got {type(value)}")

            # Check if the tuple parameters are tuples of integers
            if key in tuple_params:
                if not isinstance(value, tuple) or not all(isinstance(i, int) for i in value):
                    # In the case of xlim, it can be None
                    if key == "xlim" and value is None:
                        continue
                    raise TypeError(f"{key} must be a tuple of integers.")

            if key == "show_xticks":
                if not isinstance(value, bool):
                    raise TypeError(f"{key} must be a boolean. Got {type(value)}")

            # Check if linewidth_bars is an integer or a float
            if key == "linewidth_bars":
                if not isinstance(value, (int, float)):
                    raise TypeError(f"{key} must be an integer or a float. Got {type(value)}")

            # Check if save_dir is a string
            if key == "save_dir":
                if value is not None and not isinstance(value, str):
                    raise TypeError(f"{key} must be a string indicating path to save the plot. Got {type(value)}")

    @staticmethod
    def __minimum_distance_index(distances: Union[np.ndarray, list]) -> int:
        """
        Get the index of the minimum distance in a list of distances.

        Parameter:
            distances: `np.array` or `list`. List of distances.

        Returns:
             `int`. Index of the minimum distance.

        Raises:
            TypeError: If the distances are not a list or a numpy array.

        Notes:
            This method is private and cannot be accessed from outside the class.

        Examples:
            >>> distances = [1, 2, 3, 4, 5]
            >>> drfl = DRFL(_m=3, R=2, C=3, G=4, epsilon=0.5)
            >>> drfl.__minimum_distance_index(distances)
            0
        """
        # Check if the distances are a list
        if not isinstance(distances, list) and not isinstance(distances, np.ndarray):
            raise TypeError("distances must be a list or a numpy array")

        return int(np.argmin(distances))

    @staticmethod
    def __is_match(S1: Subsequence, S2: Union[np.ndarray, Subsequence], R: int | float) -> bool:
        """
        Check if two subsequences match by checking if the distance between them is lower than the threshold distance parameter _R.

        Parameters:
            * S1: `Subsequence`. The first subsequence.
            * S2: `np.array` or `Subsequence`. The second subsequence.
            * R: `int` or `float`. The threshold distance parameter.

        Returns:
            `bool`. `True` if the distance between the subsequences is lower than the threshold distance parameter _R, `False` otherwise.

        Raises:
            TypeError: If S1 is not an instance of `Subsequence` or S2 is not an instance of `Subsequence` or `np.ndarray`.

        Notes:
            This method is private and cannot be accessed from outside the class.

        Examples:
            >>> import numpy as np
            >>> S1 = Subsequence(instance=np.array([1, 2, 3]), date=datetime.date(2024, 1, 1), starting_point=0)
            >>> S2 = Subsequence(instance=np.array([2, 3, 4]), date=datetime.date(2024, 1, 2), starting_point=1)
            >>> drfl = DRFL(m=3, R=2, C=3, G=4, epsilon=0.5)
            >>> drfl.__is_match(S1, S2, 2)
            True

            >>> S3 = Subsequence(instance=np.array([1, 2, 6]), date=datetime.date(2024, 1, 1), starting_point=0)
            >>> drfl.__is_match(S1, S3, 2)
            False
        """

        # Check if S1 is an instance of Subsequence
        if not isinstance(S1, Subsequence):
            raise TypeError("S1 must be instance of Subsequence")

        # Check if S2 is an instance of Subsequence or np.ndarray
        if isinstance(S2, Subsequence) or isinstance(S2, np.ndarray):
            return S1.distance(S2) <= R

        raise TypeError("S2 must be instance of Subsequence or np.ndarray")

    @staticmethod
    def __is_overlap(S_i: Subsequence, S_j: Subsequence):
        """
        Check if two subsequences overlap by applying the following inequality from the paper:

        (i + p) > j or (j + q) > i

        Where:
            * i: Starting point of the first subsequence.
            * j: Starting point of the second subsequence.
            * p: Length of the first subsequence.
            * q: Length of the second subsequence.

        Parameters:
            * S_i: `Subsequence`. The first subsequence with starting point i.
            * S_j: `Subsequence`. The second subsequence with starting point j.

        Notes:
            This method is private and cannot be accessed from outside the class.

        Returns:
             `True` if they overlap, `False` otherwise.

        Raises:
            TypeError: If S_i or S_j are not instances of Subsequence.

        Examples:
            >>> import numpy as np
            >>> S1 = Subsequence(instance=np.array([1, 2, 3]), date=datetime.date(2024, 1, 1), starting_point=0)
            >>> S2 = Subsequence(instance=np.array([2, 3, 4]), date=datetime.date(2024, 1, 2), starting_point=1)
            >>> drfl = DRFL(m=3, R=2, C=3, G=4, epsilon=1.0)
            >>> drfl.__is_overlap(S1, S2)
            True

            >>> S1 = Subsequence(instance=np.array([1, 2, 3]), date=datetime.date(2024, 1, 1), starting_point=0)
            >>> S2 = Subsequence(instance=np.array([2, 3, 4]), date=datetime.date(2024, 1, 2), starting_point=4)
            >>> drfl = DRFL(m=3, R=2, C=3, G=4, epsilon=1.0)
            >>> drfl.__is_overlap(S1, S2)
            False
        """

        # Check if S_i and S_j are instances of Subsequence
        if not isinstance(S_i, Subsequence) or not isinstance(S_j, Subsequence):
            raise TypeError("S_i and S_j must be instances of Subsequence")

        start_i, p = S_i.get_starting_point(), len(S_i.get_instance())
        start_j, q = S_j.get_starting_point(), len(S_j.get_instance())
        return not ((start_i + p <= start_j) or (start_j + q <= start_i))

    @staticmethod
    def __inverse_gaussian_distance(N_target: int, N_estimated: int, sigma: float) -> float:
        """
        Compute the inverse gaussian distance between the target and estimated number of instances.

        It applies the following formula:

        1 - exp(-((N_target - N_estimated) ** 2) / sigma)

        This distance is ranged from 0 to 1, where 0 means that the target and estimated number of instances are equal and 1 means that they are different.
        Its purpose is to penalize the difference between the target and estimated number of routines in a smooth way.

        Parameters:
            * N_target: `int`. Target number of centroids.
            * N_estimated: `int`. Estimated number of centroids.
            * sigma: `float`. Standard deviation parameter for the inverse gaussian distance calculation. Lower values of sigma penalizes more the difference between the target and estimated number of centroids.

        Returns:
            `float`. The inverse gaussian distance between the target and estimated number of instances.

        Examples:
            >>> drfl = DRFL(m=3, R=2, C=3, G=4, epsilon=0.5)
            >>> drfl.__inverse_gaussian_distance(N_target=3, N_estimated=3, sigma=2)
            0
        """

        return 1 - np.exp(-((N_target - N_estimated) ** 2) / sigma)

    @staticmethod
    def __matrix_of_distances(target_centroids: list[list], estimated_centroids: list[np.ndarray]) -> np.ndarray:
        """
        Compute the matrix of distances between the target and estimated centroids.
        The distance between the target and estimated centroids is the maximum absolute difference between them.

        Parameters:
            * target_centroids: `list[list]`. Target centroids.
            * estimated_centroids: `list[np.ndarray]`. Estimated centroids.

        Returns:
            `np.array`. Matrix of distances between the target and estimated centroids.

        Examples:
            >>> target_centroids = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
            >>> estimated_centroids = [np.array([1, 2, 3]), np.array([2, 3, 4]), np.array([3, 4, 5])]
            >>> drfl.__matrix_of_distances(target_centroids, estimated_centroids)
            array([[0., 1., 2.],
                   [1., 0., 1.],
                   [2., 1., 0.]])
        """

        # Initialize the matrix of distances
        matrix = np.zeros((len(target_centroids), len(estimated_centroids)))

        # Compute the matrix of distances
        for i, target in enumerate(target_centroids):
            for j, estimated in enumerate(estimated_centroids):
                # Compute the distance between the target and estimated centroids
                matrix[i, j] = np.max(np.abs(np.array(target) - estimated))

        return matrix

    @staticmethod
    def __closest_centroids_distances(matrix_of_distances: np.ndarray) -> np.ndarray:
        """
        Compute the closest distances between the target and estimated centroids.

        Parameters:
            * matrix_of_distances: `np.ndarray`. Matrix of distances between the target and estimated centroids.

        Returns:
            `np.ndarray`. Closest distances between the target and estimated centroids.
        """

        return np.min(matrix_of_distances, axis=1)

    def _extract_subsequence(self, time_series: pd.Series, t: int) -> None:
        """
        Extract a subsequence from the time series and adds the subsequence to Sequence object.

        Parameters:
            * time_series: `pd.Series`. Temporal data.
            * t: `int`. Starting point of the subsequence.

        Raises:
            TypeError: If t is not an integer or time_series is not a pandas Series.
            ValueError: If the starting point of the subsequence is out of the time series range.

        Notes:
            This method is protected and only can be accessed from the class and subclasses

        Examples:
            >>> import pandas as pd
            >>> time_series = pd.Series([1, 2, 3, 4, 5])
            >>> time_series.index = pd.date_range(start="2024-01-01", periods=len(time_series))
            >>> drfl = DRFL(m=3, R=2, C=3, G=4, epsilon=0.5)
            >>> drfl._extract_subsequence(time_series, 0) # This property cannot be accessed from outside the class
            >>> print(drfl.__sequence)
            Sequence(
                list_sequences=[
                    Subsequence(
                        instance=np.array([1, 2, 3]),
                        date=datetime.date(2024, 1, 1),
                        starting_point=0
                    ),
                    Subsequence(
                        instance=np.array([2, 3, 4]),
                        date=datetime.date(2024, 1, 2),
                        starting_point=1
                    ),
                    Subsequence(
                        instance=np.array([3, 4, 5]),
                        date=datetime.date(2024, 1, 3),
                        starting_point=2
                    ),
                    Subsequence(
                        instance=np.array([4, 5, 6]),
                        date=datetime.date(2024, 1, 4),
                        starting_point=3
                    ),
                ]
            )
        """
        # Check if time_series is a pandas series
        self._check_type_time_series(time_series)

        # Check if t is an integer
        if not isinstance(t, int):
            raise TypeError("t must be an integer")

        # Check if t is within the range of the time series
        if t + self._m > len(time_series) or t < 0:
            raise ValueError(f"The starting point of the subsequence is out of the time series range")

        window = time_series[t:t + self._m]  # Extract the time window

        subsequence = Subsequence(instance=window.values,
                                  date=time_series.index[t],
                                  starting_point=t)  # Get the subsequence from the time window

        self.__sequence.add_sequence(subsequence)  # Add the subsequence to the sequences

    def __not_trivial_match(self, subsequence: Subsequence, cluster: Cluster, start: int, R: int | float) -> bool:
        """
        Checks if a subsequence is not a trivial match with any of the instances from the cluster.

        This method returns False if there is not a match between the
        subsequence and the centroid.
        It also returns False if there is a match between the subsequence
        and any subsequence with a starting point between the start
        parameter and the starting point of the subsequence.
        Otherwise, it returns True.

        Notes:
            This method is private and cannot be accessed from outside the class.

        Parameters:
            * subsequence: `Subsequence`. The subsequence to check.
            * cluster: `Cluster`. The cluster to check.
            * start: `int`. Starting point of the subsequence.
            * R: `int` or `float`. The threshold distance parameter.

        Returns:
            `bool`. `True` if the subsequence is not a trivial match with any of the instances from the cluster, `False` otherwise.

        Raises:
             TypeError: If subsequence is not an instance of `Subsequence` or cluster is not an instance of `Cluster`.

        Examples:
            >>> import numpy as np
            >>> S1 = Subsequence(instance=np.array([1, 2, 3]), date=datetime.date(2024, 1, 1), starting_point=0)
            >>> S2 = Subsequence(instance=np.array([2, 3, 4]), date=datetime.date(2024, 1, 2), starting_point=1)
            >>> cluster = Cluster(centroid=S2, instances=Sequence(subsequence=S2))
            >>> drfl = DRFL(m=3, R=2, C=3, G=4, epsilon=0.5)
            >>> drfl.__not_trivial_match(S1, cluster, 0, 2)
            False
            >>> drfl.__not_trivial_match(S1, cluster, 1, 2)
            True
        """

        # Check if subsequence is an instance of Subsequence and cluster is an instance of Cluster
        if not isinstance(subsequence, Subsequence) or not isinstance(cluster, Cluster):
            raise TypeError("subsequence and cluster must be instances of Subsequence and Cluster respectively")

        # Check if the subsequence is not a trivial match with any of the instances from the cluster
        if not self.__is_match(S1=subsequence, S2=cluster.centroid, R=R):
            return False

        # Check if there is a match between the subsequence and any subsequence with a starting point
        # between the start parameter and the starting point of the subsequence
        for end in cluster.get_starting_points():
            for t in reversed(range(start + 1, end)):
                # If some subsequence is a trivial match with a subsequence from the referenced
                # starting point, it returns False
                if self.__is_match(S1=subsequence, S2=self.__sequence.get_by_starting_point(t), R=R):
                    return False

        return True

    def _subgroup(self, sequence: Sequence, R: float | int, C: int, G: float | int) -> Routines:
        """
        Group the subsequences into clusters based on their magnitude and maximum absolute distance.

        The steps that follow this algorithm are:
            * Create a new cluster with the first subsequence.
            * For each subsequence, check if it is not a trivial match with any of the instances within the cluster.
            * If it is not a trivial match, append new Sequence on the instances of the cluster.
            * If it is a trivial match, create a new cluster.
            * Filter the clusters by frequency.

        Notes:
            This method is private and cannot be accessed from outside the class.

        Parameters:
            * sequences: `Sequence`. The subsequences to group into clusters.
            * R: `float` or `int`. distance threshold.
            * C: `int`. Frequency threshold.
            * G: `float` or `int`. magnitude threshold.

        Returns:
            Routines. The clusters of subsequences.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> time_series = pd.Series([1, 3, 6, 4, 2, 1, 2, 3, 6, 4, 1, 1, 3, 6, 4, 1])
            >>> time_series.index = pd.date_range(start="2024-01-01", periods=len(time_series))
            >>> drfl = DRFL(m=3, R=2, C=3, G=4, epsilon=1)
            >>> drfl.fit(time_series)
            >>> routines = drfl._subgroup()
            >>> print(routines)
            Routines(
            list_routines=[
                Cluster(
                    -Centroid: [1.33333333 3.         6.        ]
                    -Instances: [array([1, 3, 6]), array([2, 3, 6]), array([1, 3, 6])]
                    -Dates: [Timestamp('2024-01-01 00:00:00'), Timestamp('2024-01-07 00:00:00'), Timestamp('2024-01-12 00:00:00')]
                    -Starting Points: [0, 6, 11]
                ),
                Cluster(
                    -Centroid: [3. 6. 4.]
                    -Instances: [array([3, 6, 4]), array([3, 6, 4]), array([3, 6, 4])]
                    -Dates: [Timestamp('2024-01-02 00:00:00'), Timestamp('2024-01-08 00:00:00'), Timestamp('2024-01-13 00:00:00')]
                    -Starting Points: [1, 7, 12]
                ),
                Cluster(
                    -Centroid: [5.5  3.5  1.25]
                    -Instances: [array([6, 4, 2]), array([4, 2, 1]), array([6, 4, 1]), array([6, 4, 1])]
                    -Dates: [Timestamp('2024-01-03 00:00:00'), Timestamp('2024-01-04 00:00:00'), Timestamp('2024-01-09 00:00:00'), Timestamp('2024-01-14 00:00:00')]
                    -Starting Points: [2, 3, 8, 13]
                )]
            )
        """

        routines = Routines()

        # Iterate through all the subsequences
        for i in range(len(sequence)):
            # Check if the magnitude of the subsequence is greater than G and the minimum of the sequence is greater than L
            if sequence[i].magnitude() >= G and sequence[i].inverse_magnitude() >= self._L:
                if routines.is_empty():  # Initialize first cluster if its empty
                    # Create a cluster from the first subsequence
                    routines.add_routine(Cluster(centroid=sequence[i].get_instance(),
                                                 instances=Sequence(subsequence=sequence[i])))
                    continue  # Continue to the next iteration

                # Estimate all the distances between the subsequence and all the centroids of the clusters
                distances = [sequence[i].distance(routines[j].centroid) for j in range(len(routines))]

                # Get the index of the minimum distance to the centroid
                j_hat = self.__minimum_distance_index(distances)

                # Check if the subsequence is not a trivial match with any of the instances within the cluster
                # if self.__not_trivial_match(subsequence=self.sequence[i], cluster=routines[j_hat], start=i, _R=_R):
                if self.__is_match(S1=sequence[i], S2=routines[j_hat].centroid, R=R):
                    routines[j_hat].add_instance(sequence[i])  # Append new Sequence on the instances of Bm_j
                    routines[j_hat].update_centroid()  # Update center of the cluster

                else:
                    # create a new cluster//routine
                    new_cluster = Cluster(centroid=sequence[i].get_instance(),
                                          instances=Sequence(subsequence=sequence[i]))
                    routines.add_routine(new_cluster)  # Add the new cluster to the routines

        # Filter by frequency
        to_drop = [k for k in range(len(routines)) if len(routines[k]) < C]
        filtered_routines = routines.drop_indexes(to_drop)

        return filtered_routines

    def __overlapping_test(self, cluster1: Cluster, cluster2: Cluster, epsilon: float) -> tuple[bool, bool]:
        """
        Test and handle overlapping clusters by determining the significance of their overlap.

        Overlapping clusters are analyzed to decide if one, both, or none should be kept based on the overlap
        percentage and the clusters' characteristics. This determination is crucial for maintaining the
        quality and interpretability of the detected routines. The method employs a two-step process: first,
        it calculates the number of overlapping instances between the two clusters; then, based on the overlap
        percentage and the clusters' properties (e.g., size and magnitude), it decides which cluster(s) to retain.

        Parameters:
            * cluster1: `Cluster`. The first cluster involved in the overlap test.
            * cluster2: `Cluster`. The second cluster involved in the overlap test.
            * epsilon: `float`. A threshold parameter that defines the minimum percentage of overlap required for considering an overlap significant. Values range from 0 to 1, where a higher value means a stricter criterion for significance.

        Returns:
            * tuple[bool, bool]: A tuple containing two boolean values. The first value indicates whether
                                 cluster1 should be kept (True) or discarded (False). Similarly, the second
                                 value pertains to cluster2.


        Overview of the Method's Logic:
            * Calculate the number of instances in cluster1 that significantly overlap with any instance in cluster2.
            * determine the significance of the overlap based on the '_epsilon' parameter: if the number of overlaps exceeds '_epsilon' times the smaller cluster's size, the overlap is considered significant.
            * In case of significant overlap, compare the clusters based on their size and the cumulative magnitude of their instances. The cluster with either a larger size or a greater cumulative magnitude (in case of a size tie) is preferred.
            * Return a tuple indicating which clusters should be kept. If the overlap is not significant, both clusters may be retained.

        Note:
            * This method relies on private helper methods to calculate overlaps and compare cluster properties.
            * The method does not modify the clusters directly but provides guidance on which clusters to keep or discard.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> S1 = Subsequence(instance=np.array([1, 2, 3]), date=datetime.date(2024, 1, 1), starting_point=0)
            >>> S2 = Subsequence(instance=np.array([2, 3, 4]), date=datetime.date(2024, 1, 2), starting_point=1)
            >>> cluster1 = Cluster(centroid=S1, instances=Sequence(subsequence=S1))
            >>> cluster2 = Cluster(centroid=S2, instances=Sequence(subsequence=S2))
            >>> drfl = DRFL(m=3, R=2, C=3, G=4, epsilon=1.0)
            >>> drfl.__overlapping_test(cluster1, cluster2, 0.5)
            (True, False)
        """

        N = 0  # Initialize counter for number of overlaps

        # Iterate through all instances in cluster1
        for S_i in cluster1.get_sequences():
            # Convert instance to Subsequence if needed for overlap checks
            for S_j in cluster2.get_sequences():
                # Check for overlap between S_i and S_j
                if self.__is_overlap(S_i, S_j):
                    N += 1  # Increment overlap count
                    break  # Break after finding the first overlap for S_i

        # Calculate the minimum length of the clusters to determine significance of overlap
        min_len = min(len(cluster1), len(cluster2))

        # Determine if the overlap is significant based on _epsilon and the minimum cluster size
        if N > epsilon * min_len:

            # Calculate cumulative magnitudes for each cluster to decide which to keep
            mag_cluster1 = cluster1.cumulative_magnitude()
            mag_cluster2 = cluster2.cumulative_magnitude()

            # Keep the cluster with either more instances or, in a tie, the greater magnitude
            if len(cluster1) > len(cluster2) or (len(cluster1) == len(cluster2) and mag_cluster1 > mag_cluster2):
                return True, False
            else:
                return False, True
        else:
            # If overlap is not significant, propose to keep both clusters
            return True, True

    def __obtain_keep_indices(self, epsilon: float) -> list[int]:
        """
        Obtain the indices of the clusters to keep based on the overlap test.

        Parameters:
            epsilon: `float`. A threshold parameter that defines the minimum percentage of overlap required for considering an overlap significant. Values range from 0 to 1, where a higher value means a stricter criterion for significance.

        Returns:
            `list[int]`. The indices of the clusters to keep.

        Raises:
             ValueError: If _epsilon is not between 0 and 1.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> S1 = Subsequence(instance=np.array([1, 2, 3]), date=datetime.date(2024, 1, 1), starting_point=0)
            >>> S2 = Subsequence(instance=np.array([2, 3, 4]), date=datetime.date(2024, 1, 2), starting_point=4)
            >>> cluster1 = Cluster(centroid=S1, instances=Sequence(subsequence=S1))
            >>> cluster2 = Cluster(centroid=S2, instances=Sequence(subsequence=S2))
            >>> drfl = DRFL(m=3, R=2, C=3, G=4, epsilon=1.0)
            >>> drfl.__obtain_keep_indices(1)
            [0, 1]

            >>> S1 = Subsequence(instance=np.array([1, 2, 3]), date=datetime.date(2024, 1, 1), starting_point=0)
            >>> S2 = Subsequence(instance=np.array([2, 3, 4]), date=datetime.date(2024, 1, 2), starting_point=1)
            >>> cluster1 = Cluster(centroid=S1, instances=Sequence(subsequence=S1))
            >>> cluster2 = Cluster(centroid=S2, instances=Sequence(subsequence=S2))
            >>> drfl = DRFL(m=3, R=2, C=3, G=4, epsilon=0.5)
            >>> drfl.__obtain_keep_indices(0.5)
            [1]
        """

        if epsilon < 0 or epsilon > 1:
            raise ValueError("_epsilon must be between 0 and 1")

        # Prepare to test and handle overlapping clusters
        keep_indices = set(range(len(self.__routines)))  # Initially, assume all clusters are to be kept

        for i in range(len(self.__routines) - 1):
            for j in range(i + 1, len(self.__routines)):
                if i in keep_indices and j in keep_indices:  # Process only if both clusters are still marked to keep
                    keep_i, keep_j = self.__overlapping_test(self.__routines[i], self.__routines[j], epsilon)

                    # Update keep indices based on OLTest outcome
                    if not keep_i:
                        keep_indices.remove(i)
                    if not keep_j:
                        keep_indices.remove(j)

        return list(keep_indices)

    def fit(self, time_series: pd.Series) -> None:
        """
        Fits the time series data to the `DRFL` algorithm to discover routines.

        This method preprocesses the time series data, extracts subsequences, groups them into clusters, and finally filters and handles overlapping clusters to discover and refine routines.

        Parameters:
             time_series: `pd.Series`. The time series data to analyze. It should be a `pandas Series` object with a `DatetimeIndex`.

        Raises:
             TypeError: If the input time series is not a `pandas Series` or if its index is not a `DatetimeIndex`.

        Examples:
            >>> import pandas as pd
            >>> time_series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            >>> time_series.index = pd.date_range(start="2024-01-01", periods=len(time_series))
            >>> drfl = DRFL(m=3, R=2, C=3, G=4, epsilon=0.5)
            >>> drfl.fit(time_series)
            >>> print(drfl.routines)
        """

        self._check_type_time_series(time_series)
        self.time_series = time_series

        # Set the already_fitted attribute to True
        self.__already_fitted = True

        for i in range(len(self.time_series) - self._m + 1):
            self._extract_subsequence(self.time_series, i)

        # Group the subsequences into clusters based on their magnitude and
        # maximum absolute distance and filter the clusters based on their frequency
        self.__routines = self._subgroup(sequence=self.__sequence, R=self._R, C=self._C, G=self._G)

        # Obtain the indices of the clusters to keep based on the overlap test
        keep_indices = self.__obtain_keep_indices(self._epsilon)

        # Filter self.routines to keep only those clusters marked for keeping
        if len(self.__routines) > 0:
            to_drop = [k for k in range(len(self.__routines)) if k not in keep_indices]
            self.__routines = self.__routines.drop_indexes(to_drop)

        if len(self.__routines) == 0:
            warnings.warn("No routines have been discovered", UserWarning)

    def estimate_distance(self, target_centroids: list[list], alpha: float, sigma: float) -> float:
        """
        Estimate the distance between the target centroids and the estimated centroids.
        The distance is a combination of the penalization
        of detecting a distinct number of routines and the normalized distance between the target and estimated centroids.
        Applies the following formula:

        alpha * penalization + (1 - alpha) * normalized_distance

        Where penalization is the inverse gaussian distance between the target and estimated number of instances,
        and normalized_distance is the sum of the closest distances between the target and estimated centroids
        divided by the total sum of the distances.

        The result is a distance value ranged from 0 to 1,
        where 0 means that the target and estimated centroids are equal and 1 means that they are different.

        Parameters:
            * target_centroids: `list[list]`. Target centroids.
            * alpha: `float`. Weight parameter to balance the penalization and normalized distance.
            * sigma: `float`. Standard deviation parameter for the inverse gaussian distance calculation.

        Returns:
            `float`. The distance between the target and estimated centroids ranged from 0 to 1.

        Raises:
            RuntimeError: If the model has not been fitted yet.
            TypeError: If target_centroids is not a list of lists.
            ValueError: If alpha is not between 0 and 1 or sigma is not greater than 1.

        Examples:
            >>> import numpy as np
            >>> target_centroids = [[4 / 3, 3, 6], [3, 6, 4], [6, 4, 4 / 3]]
            >>> time_series = pd.Series([1, 3, 6, 4, 2, 1, 2, 3, 6, 4, 1, 1, 3, 6, 4, 1])
            >>> time_series.index = pd.date_range(start="2024-01-01", periods=len(time_series))
            >>> drfl = DRFL(m=3, R=1, C=3, G=4, epsilon=1)
            >>> drfl.fit(time_series)
            >>> dist = drfl.estimate_distance(target_centroids, alpha=0.5, sigma=3)
            >>> print(dist)
            0.0
        """

        # Check if the model has been fitted
        if not self.__already_fitted:
            raise RuntimeError("The model has not been fitted yet. Please call the fit method before using this method")

        # Check if there are routines to compare
        if self.__routines.is_empty():
            warnings.warn("No routines have been discovered", UserWarning)
            return np.nan

        # Estimate the penalization of detecting a distinct number of routines
        N_target, N_estimated = len(target_centroids), len(self.__routines)

        # estimate the penalization of detecting a distinct number of routines
        penalization = self.__inverse_gaussian_distance(N_target, N_estimated, sigma)

        # calculate the matrix of distances between the target centroids and the estimated centroids
        matrix_of_distances = self.__matrix_of_distances(target_centroids, self.__routines.get_centroids())

        # calculate the closest distances between the target centroids and the estimated centroids
        closest_distances = self.__closest_centroids_distances(matrix_of_distances)

        # Normalization of the distances
        total_sum_matrix = np.sum(matrix_of_distances)
        total_sum_closest = np.sum(closest_distances)

        # Avoid division by zero
        if total_sum_matrix == 0:
            normalized_distance = 0
        else:
            normalized_distance = total_sum_closest / total_sum_matrix

        return alpha * penalization + (1 - alpha) * normalized_distance

    def show_results(self) -> None:
        """
        Displays the discovered routines after fitting the model to the time series data.

        This method prints out detailed information about each discovered routine, including the centroid of each cluster, the subsequence instances forming the routine, and the dates/times these routines occur.

        Note:
            This method should be called after the `fit` method to ensure that routines have been discovered and are ready to be displayed.
        """

        if not self.__already_fitted:
            raise RuntimeError("The model has not been fitted yet. Please call the fit method before using this method")

        print("Routines detected: ", len(self.__routines))
        print("_" * 50)
        for i, b in enumerate(self.__routines):
            print(f"Centroid {i + 1}: {b.centroid}")
            print(f"Routine {i + 1}: {b.get_sequences().get_subsequences()}")
            print(f"Date {i + 1}: {b.get_dates()}")
            print(f"Starting Points {i + 1}: ", b.get_starting_points())
            print("\n", "-" * 50, "\n")

    def get_results(self) -> Routines:
        """
        Returns the discovered routines as a `Routines` object.

        After fitting the model to the time series data, this method can be used to retrieve the discovered routines, encapsulated within a `Routines` object, which contains all the clusters (each representing a routine) identified by the algorithm.

        Returns:
             `Routines`. The discovered routines as a `Routines` object.

        Note:
            The `Routines` object provides methods and properties to further explore and manipulate the discovered routines.
        """

        if not self.__already_fitted:
            raise RuntimeError("The model has not been fitted yet. Please call the fit method before using this method")

        return self.__routines

    def plot_results(self, title_fontsize: int = 20,
                     xticks_fontsize: int = 20, yticks_fontsize: int = 20,
                     labels_fontsize: int = 20, figsize: tuple[int, int] = (30, 10),
                     text_fontsize: int = 20, linewidth_bars: int = 1.5,
                     xlim: Optional[tuple[int, int]] = None,
                     save_dir: Optional[str] = None) -> None:

        """
        This method uses matplotlib to plot the results of the algorithm. The plot shows the time series data with vertical dashed lines indicating the start of each discovered routine. The color of each routine is determined by the order in which they were discovered, and a legend is displayed to identify each routine.

        Parameters:
            * title_fontsize: `int` (default is 20). Size of the title plot.
            * xticks_fontsize: `int` (default is 20). Size of the xticks.
            * yticks_fontsize: `int (default is 20)`. Size of the yticks.
            * labels_fontsize: `int` (default is 20). Size of the labels.
            * figsize: `tuple[int, int]` (default is (30, 10)). Size of the figure.
            * text_fontsize: `int` (default is 20). Size of the text.
            * linewidth_bars: `int` (default is 1.5). Width of the bars in the plot.
            * xlim: `tuple[int, int]` (default is None). Limit of the x axis with starting points.
            * save_dir: `str` (default is None). Directory to save the plot.

        Notes:
           This method has to be executed after the fit method to ensure that routines have been discovered and are ready to be displayed.
        """

        args = locals()
        self._check_plot_params(**args)

        # Check if the model has been fitted
        if not self.__already_fitted:
            raise RuntimeError("The model has not been fitted yet. Please call the fit method before using this method")

        # Generate a color map for the routines
        base_colors = cm.rainbow(np.linspace(0, 1, len(self.__routines)))

        # Convert the time series data to a numpy array for easier manipulation
        ts = np.array(self.time_series)

        # Create a new figure with the specified size
        plt.figure(figsize=figsize)

        # Get the number of routines and the maximum value in the time series
        N_rows = len(self.__routines)
        maximum = max(ts)

        # if xlim is not provided, set the limits of the x-axis to the range of the time series
        xlim = xlim or (0, len(ts) - 1)

        # Get the starting points of each routine
        start_points = [cluster.get_starting_points() for cluster in self.__routines]

        # For each routine, create a subplot and plot the routine
        for row, routine in enumerate(start_points):
            plt.subplot(N_rows, 1, row + 1)

            # Initialize the color of each data point in the time series as gray
            colors = ["gray"] * len(ts)

            # Set the title and x-label of the subplot
            plt.title(f'Routine {row + 1}', fontsize=title_fontsize)
            plt.xlabel("Starting Points", fontsize=labels_fontsize)

            # For each starting point in the routine, plot a vertical line and change the color of the data points in the routine
            for sp in routine:
                if xlim[0] <= sp <= xlim[1]:
                    plt.axvline(x=sp, color=base_colors[row], linestyle="--")
                    for j in range(self._m):
                        if sp + j <= xlim[1]:
                            plt.text(sp + j - 0.05, self.time_series[sp + j] - 0.8, f"{ts[sp + j]}",
                                     fontsize=text_fontsize, backgroundcolor="white", color=base_colors[row])
                            colors[sp + j] = base_colors[row]

            # Plot the time series data as a bar plot
            plt.bar(x=np.arange(0, len(ts)), height=ts, color=colors, edgecolor="black", linewidth=linewidth_bars)

            # Set the ticks on the x-axis
            plt.xticks(ticks=np.arange(xlim[0], xlim[1] + 1),
                       labels=np.arange(xlim[0], xlim[1] + 1),
                       fontsize=xticks_fontsize)

            plt.yticks(fontsize=yticks_fontsize)

            # Plot a horizontal line at the magnitude threshold
            plt.axhline(y=self._G, color="red", linestyle="--")

            # Set the limits of the x-axis and y-axis
            plt.xlim(xlim[0] - 0.5, xlim[1] + 0.5)
            plt.ylim(0, maximum + 1)

            # Adjust the layout of the plot
            plt.tight_layout()

        # If a directory is provided, save the figure to the directory
        if save_dir:
            plt.savefig(save_dir)

        # Display the plot
        plt.show()


class ParallelSearchDRFL:
    """
    Class to perform a parallel search of the best parameters for the DRFL algorithm using multithreading.

    This class allows the user to search for the best parameters for the DRFL algorithm using a grid search approach.

    Parameters:
        * n_jobs: `int`. The number of parallel jobs to run.
        * alpha: `int | float`. Rate of penalization.
        * sigma: `int | float`. Sigma parameter for the variance in the inverse Gaussian distance.
        * param_grid: `dict`. Dictionary with parameters names (`_m`, `_R`, `_C`, `_G`, `_epsilon`) as keys and lists of their values to try, representing the parameters from the DRFL algorithm.

    Attributes:
    -------
        * n_jobs: `int`. The number of parallel jobs to run.
        * alpha: `int | float`. Rate of penalization.
        * sigma: `int | float`. Sigma parameter for the variance in the inverse Gaussian distance.
        * param_grid: `dict`. Dictionary with parameters names (`_m`, `_R`, `_C`, `_G`, `_epsilon`) as keys and lists of their values to try, representing the parameters from the DRFL algorithm.

    Methods:
    -------
        * fit(time_series: pd.Series, target_centroids: list[list]): Fit the DRFL algorithm to the time series data using multiple sets of parameters in parallel.
        * best_params(): Get the best parameters found during the search.
        * cv_results(): Get the results of the search.


    Examples:
    --------
        >>> import pandas as pd
        >>> time_series = pd.Series([1, 3, 6, 4, 2, 1, 2, 3, 6, 4, 1, 1, 3, 6, 4, 1])
        >>> time_series.index = pd.date_range(start="2024-01-01", periods=len(time_series))
        >>> param_grid = {
        ...     "_m": [3],
        ...     "_R": [1, 2, 3, 4, 5, 6],
        ...     "_C": [1, 2, 3, 4, 5, 6],
        ...     "_G": [1, 2, 3, 4, 5, 6],
        ...     "_epsilon": [0.5, 1],
        ... }
        >>> search = ParallelSearchDRFL(n_jobs=2, alpha=0.5, sigma=3, param_grid=param_grid)
        >>> search.fit(time_series)
        >>> print(search.best_params())
        >>> {'_m': 3, '_R': 1, '_C': 3, '_G': 4, '_epsilon': 1}
    """

    def __init__(self, n_jobs: int, alpha: Union[int, float], sigma: Union[int, float], param_grid: dict):
        """
        Initialize the ParallelDRFL object with a parameter grid for the DRFL algorithm and the number of jobs for parallel processing.

        Parameters:
            * n_jobs: `int`. The number of parallel jobs to run.
            * alpha: `int | float`. Rate of penalization.
            * sigma: `int | float`. Sigma parameter for the variance in the inverse Gaussian distance.
            * param_grid: `dict`. Dictionary with parameters names (`_m`, `_R`, `_C`, `_G`, `_epsilon`) as keys and lists of their values to try, representing the parameters from the DRFL algorithm.

        Raises:
            TypeError: If the parameter grid is not a dictionary.
            ValueError: If the parameter grid is empty or if some parameter is not valid or has an invalid value.
        """
        # Check the validity of the parameters
        self.__check_validity_params(alpha, sigma, param_grid)

        # Set the attributes
        self.__n_jobs: int = n_jobs
        self.__alpha: Union[int, float] = alpha
        self.__sigma: Union[int, float] = sigma
        self.__param_grid: dict = param_grid

        # Initialize the results attribute and the fitted attribute
        self.__results: list[dict] = []
        self.__fitted: bool = False

    @staticmethod
    def __check_m_param(m: int):
        """
        Check the validity of the _m parameter.

        Parameters:
            * m: `int`. The _m parameter to check.

        Raises:
            TypeError: If _m is not an integer.
            ValueError: If _m is not greater than 1.
        """

        if not isinstance(m, int):
            raise TypeError("_m must be an integer")

        if m < 2:
            raise ValueError("_m must be greater or equal than 2")

    @staticmethod
    def __check_R_or_G_param(param: list[Union[int, float]]):
        """
        Check the validity of the _R parameter.

        Parameters:
            * _R: `list[int | float]`. The _R parameter to check.

        Raises:
            TypeError: If _R is not a list.
            ValueError: If _R is empty or if some value in _R is not greater than one.
        """

        if not isinstance(param, list):
            raise TypeError("_R must be a list")

        if len(param) == 0:
            raise ValueError("_R cannot be empty")

        for value in param:
            if value < 1:
                raise ValueError("_R must be greater or equal than 1")

    @staticmethod
    def __check_C_param(C: list[int]):
        """
        Check the validity of the _C parameter.

        Parameters:
            * C: `list[int]`. The _C parameter to check.

        Raises:
            TypeError: If _C is not a list.
            ValueError: If _C is empty or if some value in _C is not greater than one.
        """

        if not isinstance(C, list):
            raise TypeError("_C must be a list")

        if len(C) == 0:
            raise ValueError("_C cannot be empty")

        for value in C:
            if value < 1 or not isinstance(value, int):
                raise ValueError("_C must be an integer greater or equal than 1")

    @staticmethod
    def __check_epsilon_param(epsilon: list[float | int]):
        """
        Check the validity of the _epsilon parameter.

        Parameters:
            * epsilon: `list[float | int]`. The _epsilon parameter to check.

        Raises:
            TypeError: If _epsilon is not a list.
            ValueError: If _epsilon is empty or if some value in _epsilon is not between 0 and 1.
        """

        if not isinstance(epsilon, list):
            raise TypeError("_epsilon must be a list")

        if len(epsilon) == 0:
            raise ValueError("_epsilon cannot be empty")

        for value in epsilon:
            if value < 0 or value > 1:
                raise ValueError("_epsilon must be between 0 and 1")

    def __check_validity_params(self, alpha: Union[int, float], sigma: Union[int, float], param_grid: dict):
        """
        Check the validity of the parameter grid.

        Parameters:
            * param_grid: `dict`. The parameter grid to check.

        Raises:
            TypeError: If the parameter grid is not a dictionary.
            ValueError: If some parameter is not valid or has an invalid value.
        """

        # Check if alpha and sigma are valid
        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be between 0 and 1")

        if sigma < 1:
            raise ValueError("sigma must be greater or equal than 1")

        # Check if the parameter grid is a dictionary or if it is empty
        if not isinstance(param_grid, dict):
            raise TypeError("param_grid must be a dictionary")

        if not param_grid:
            raise ValueError("param_grid cannot be empty")

        # Check if the parameters are valid
        valid_params = ['_m', '_R', '_C', '_G', '_epsilon']
        for key in param_grid.keys():
            if key not in valid_params:
                raise ValueError(f"Invalid parameter: {key}. Valid parameters are: {valid_params}")

            if key == "_m":
                self.__check_m_param(param_grid[key])

            if key == "_R" or key == "_G":
                self.__check_R_or_G_param(param_grid[key])

            if key == "_C":
                self.__check_C_param(param_grid[key])

            if key == "_epsilon":
                self.__check_epsilon_param(param_grid[key])

    @staticmethod
    def fit_single_instance(params):
        """
        Fit a single instance of the DRFL algorithm with a given set of parameters.

        Parameters:
            params (dict): A dictionary containing the parameters for a single DRFL instance.

        Returns:
            A dictionary containing the parameters used and the results of the DRFL fitting process.
        """

        m, R, C, G, epsilon, alpha, sigma, time_series, target_centroids = params
        drfl = DRFL(m=m, R=R, C=C, G=G, epsilon=epsilon)
        drfl.fit(time_series)
        estimated_distance = drfl.estimate_distance(target_centroids, alpha=alpha, sigma=sigma)
        return {"_m": m, "_R": R, "_C": C, "_G": G, "_epsilon": epsilon, "estimated_distance": estimated_distance}

    def fit(self, time_series: pd.Series, target_centroids: list[list]):
        """
        Fit the DRFL algorithm to the time series data using multiple sets of parameters in parallel.

        Parameters:
            time_series (pd.Series): The time series data to analyze.
            target_centroids (list[list]): List of target centroids to compare with the discovered routines.
        """

        # set the fitted parameter to true
        self.__fitted = True

        # Prepare the list with all combinations of parameters to fit the DRFL instances
        all_params = list(product(
            [self.__param_grid.get('_m', [3])],
            self.__param_grid.get('_R', [2]),
            self.__param_grid.get('_C', [3]),
            self.__param_grid.get('_G', [4]),
            self.__param_grid.get('_epsilon', [1]),
            [self.__alpha],
            [self.__sigma],
            [time_series],
            [target_centroids]
        ))

        # Use ProcessPoolExecutor to fit DRFL instances in parallel
        with ProcessPoolExecutor(max_workers=self.__n_jobs) as executor:
            results = list(executor.map(self.fit_single_instance, all_params))

        self.__results = results

    def cv_results(self) -> pd.DataFrame:
        """
        Return the cross-validation results after fitting the DRFL instances.

        Returns:
            A pandas DataFrame containing the results of the parallel search sorted by distance.

        Raises:
            RuntimeError: If the model has not been fitted yet.

        Examples:
            >>> import pandas as pd

            >>> time_series = pd.Series([1, 3, 6, 4, 2, 1, 2, 3, 6, 4, 1, 1, 3, 6, 4, 1])
            >>> time_series.index = pd.date_range(start="2024-01-01", periods=len(time_series))
            >>> param_grid = {
            ...     "_m": [3],
            ...     "_R": [1, 2, 3, 4, 5, 6],
            ...     "_C": [1, 2, 3, 4, 5, 6],
            ...     "_G": [1, 2, 3, 4, 5, 6],
            ...     "_epsilon": [0.5, 1],
            ... }
            >>> search = ParallelSearchDRFL(n_jobs=2, alpha=0.5, sigma=3, param_grid=param_grid)
            >>> search.fit(time_series)
            >>> print(search.cv_results())
            ... _m _R  _C  _G  _epsilon  estimated_distance
            ... 3 1  3  4   1.0     0.0
            ... 3 1  2  5   1.0     0.0
            ... 3 2  1  5   1.0     0.0
            ... 3 2  2  5   1.0     0.0
            ...
        """

        # Check if the model has been fitted
        if not self.__fitted:
            raise RuntimeError("The model has not been fitted yet. Please call the fit method before using this method")

        # Sort the results by estimated distance
        results = pd.DataFrame(self.__results).sort_values(by="estimated_distance")
        return pd.DataFrame(results)

    def best_params(self) -> dict:
        """
        Return the best parameters found during the search.

        Returns:
            `dict`. A dictionary containing the best parameters found during the search.

        Raises:
            RuntimeError: If the model has not been fitted yet.

        Examples:
            >>> import pandas as pd
            >>> time_series = pd.Series([1, 3, 6, 4, 2, 1, 2, 3, 6, 4, 1, 1, 3, 6, 4, 1])
            >>> time_series.index = pd.date_range(start="2024-01-01", periods=len(time_series))
            >>> param_grid = {
            ...     "_m": [3],
            ...     "_R": [1, 2, 3, 4, 5, 6],
            ...     "_C": [1, 2, 3, 4, 5, 6],
            ...     "_G": [1, 2, 3, 4, 5, 6],
            ...     "_epsilon": [0.5, 1],
            ... }
            >>> search = ParallelSearchDRFL(n_jobs=2, alpha=0.5, sigma=3, param_grid=param_grid)
            >>> search.fit(time_series)
            >>> print(search.best_params())
            >>> {'_m': 3, '_R': 1, '_C': 3, '_G': 4, '_epsilon': 1}
        """

        if not self.__fitted:
            raise RuntimeError("The model has not been fitted yet. Please call the fit method before using this method")

        results = self.cv_results()
        return results.iloc[0].to_dict()


class DRGS(DRFL):
    """
    Class to perform the DRGS algorithm.
    The DRGS algorithm is an extension of the DRFL algorithm
    that discovers routines in time series data by growing subsequences from the left and right sides of the discovered routines.

    Parameters:
    __________
        * ``length_range: tuple[int, int]``. tuple that indicates the range from the minimum length of subsequences to the maximum
        * ``_R: float | int``. distance threshold
        * ``_C: int``. frequency threshold
        * ``_G: float | int``. magnitude threshold
        * ``_epsilon: float``. overlap parameter

    Public methods:
    ___________
        * ``fit(time_series: pd.Series)``: method that applies the DRGS algorithm to the time_series input data
        * ``get_results() -> HierarchyRoutine``: returns the estimated hierarchical routines as a `HierarchyRoutine` object
        * ``show_results()``: shows in a friendly way, which ones are the routines estimated
        * ``plot_results(title_fontsize: int = 20, xticks_fontsize: int = 20, yticks_fontsize: int = 20, labels_fontsize: int = 20, figsize: tuple[int, int] = (30, 10),  text_fontsize: int = 20, linewidth_bars: int = 1.5, xlim: Optional[tuple[int, int]] = None,  save_dir: Optional[str] = None)``: user-friendly method to visualize the results of the routine detection

    Examples:
    ________
        >>> import pandas as pd
        >>> time_series = pd.Series([1, 3, 6, 4, 2, 1, 2, 3, 6, 4, 1, 1, 3, 6, 4, 1])
        >>> time_series.index = pd.date_range(start="2024-01-01", periods=len(time_series))
        >>> drgs = DRGS(length_range=(3, 8), R=2, C=3, G=4, epsilon=0.5)
        >>> drgs.fit(time_series)
        >>> hierarchical_routines = drgs.get_results()

        >>> drgs.show_results()

        >>> drgs.plot_results()


    """

    def __init__(self, length_range: tuple[int, int], R: Union[float, int], C: int, G: Union[float, int],
                 epsilon: float, L: Union[int, float] = 0) -> None:
        """
        Initialize the DRGS object with the parameters for the DRGS algorithm

        Parameters:
            * length_range: `tuple[int, int]`. tuple that indicates the range from the minimum length of subsequences to the maximum
            * R: `float | int`. distance threshold
            * C: `int`. frequency threshold
            * G: `float | int`. magnitude threshold
            * epsilon: `float`. overlap parameter
            * L: `int | float`. length of the subsequence

        Raises:

        """
        # Check the validity of the parameters
        params = locals()
        super()._check_params(**params)

        # Set the attributes of the DRFL object by default
        super().__init__(m=length_range[0], R=R, C=C, G=G, epsilon=epsilon, L=L)

        # Set the attributes of the DRGS object
        self.__length_range = length_range
        self.__hierarchical_routines = HierarchyRoutine()
        self.time_series: pd.Series

        # Set the already_fitted attribute to False
        self.__already_fitted = False

    @staticmethod
    def __union_routines(left: Routines, right: Routines) -> Routines:
        """
        Union of two routines.

        Returns:
            `Routines`. The union of the two routines in one routine with the clusters combined.

        Examples:
            >>> left = Routines(Cluster(centroid=np.array([1, 2, 3]), instances=Sequence(Subsequence(np.array([1, 2, 3]), date=datetime.date(2024, 1, 1), starting_point=0))))
            >>> right = Routines(Cluster(centroid=np.array([3, 2, 1]), instances=Sequence(Subsequence(np.array([3, 2, 1]), date=datetime.date(2024, 1, 1), starting_point=0))))
            >>> union = DRGS.__union_routines(left, right)
            >>> print(union)
            Routines(
                list_routines=[
                    Cluster(
                        - centroid=np.array([1, 2, 3])
                        - instances=[[1, 2, 3]]
                        - date=datetime.date(2024, 1, 1),
                        - starting_point=[0]
                        ),
                    Cluster(
                        - centroid=np.array([3, 2, 1])
                        - instances=[[3, 2, 1]]
                        - date=datetime.date(2024, 1, 1),
                        - starting_point=[0]
                        )
                    ]
                )
        """

        return left + right

    def __grow_from_left(self, sequence: Sequence) -> Sequence:
        """
        Grow the sequence from the left side.
        This method detects the length of subsequences and grows them from the left to right side.

        Parameters:
            * sequence: `Sequence`. The sequence to grow from the left side.

        Returns:
            `Sequence`. The sequence grown from the left side.

        Examples:
            >>> import pandas as pd
            >>> time_series = pd.Series([1, 3, 6, 4, 2, 1, 2, 3, 6, 4, 1, 1, 3, 6, 4, 1])
            >>> time_series.index = pd.date_range(start="2024-01-01", periods=len(time_series))
            >>> drgs = DRGS(length_range=(3, 8), R=2, C=3, G=4, epsilon=0.5)
            >>> drgs.fit(time_series)

            >>> sequence = Sequence(Subsequence(np.array([1, 3, 6, 4]), datetime.date(2024, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([2, 3, 6, 4]), datetime.date(2024, 1, 7), 6))
            >>> sequence.add_sequence(Subsequence(np.array([1, 3, 6, 4]), datetime.date(2024, 1, 12), 11))

            >>> sequence_left = drgs._DRGS__grow_from_left(sequence)
            >>> print(sequence_left)
            Sequence(
            list_sequences=[
                Subsequence(
                    - instance = [1, 3, 6, 4]
                    - date = 2024-1-1
                    - starting_point = 0
                ),
                Subsequence(
                    - instance = [2, 3, 6, 4]
                    - date = 2024-1-7
                    - starting_point = 6
                ),
                Subsequence(
                    - instance = [1, 3, 6, 4]
                    - date = 2024-1-12
                    - starting_point = 11
                )
            ]
        """
        # Increment the length of subsequences
        m_next = sequence.length_subsequences + 1

        # Extract the components from the sequence
        _, dates, starting_points = sequence.extract_components(flatten=True)
        sequence_left = Sequence()

        # Iterate over the sequence
        for i in range(len(sequence)):
            # Check if the starting point plus the length of the subsequence is less than the length of the time series
            if starting_points[i] + m_next < len(self.time_series):
                instance = self.time_series[starting_points[i]:starting_points[i] + m_next].values
                sequence_left.add_sequence(
                    Subsequence(instance=instance, date=dates[i], starting_point=starting_points[i])
                )

        return sequence_left

    def __grow_from_right(self, sequence: Sequence) -> Sequence:
        m_next = sequence.length_subsequences + 1
        _, dates, starting_points = sequence.extract_components(flatten=True)
        sequence_right = Sequence()
        for i in range(len(sequence)):
            if starting_points[i] - 1 > 0:
                instance = self.time_series[starting_points[i] - 1:starting_points[i] + m_next - 1].values
                sequence_right.add_sequence(
                    Subsequence(instance=instance,
                                date=self.time_series.index[starting_points[i] - 1],
                                starting_point=starting_points[i] - 1)
                )

        return sequence_right

    def __overlap_remove(self, routines: Routines) -> Routines:
        pass

    def __execute_drfl(self, time_series: pd.Series, m: int) -> Routines:
        super().__init__(m=m, R=self._R, C=self._C, G=self._G, epsilon=self._epsilon, L=self._L)
        super().fit(time_series)
        return super().get_results()

    def fit(self, time_series: pd.Series):
        super()._check_type_time_series(time_series)
        self.time_series = time_series

        # Initialization
        init_routine = self.__execute_drfl(self.time_series, self.__length_range[0])
        actual_length = self.__length_range[0] + 1

        if init_routine.is_empty():
            warnings.warn("No routines have been discovered", UserWarning)
            return self.__hierarchical_routines

        self.__hierarchical_routines.add_routine(init_routine)

        # iterate over the range of lengths
        while actual_length <= self.__length_range[1] and not \
                self.__hierarchical_routines[actual_length - 1].is_empty():

            routines_l_k = Routines()
            for k, cluster in enumerate(self.__hierarchical_routines[actual_length - 1]):
                instances = cluster.get_sequences()

                left_grow = self.__grow_from_left(instances)
                right_grow = self.__grow_from_right(instances)

                left_routines = super()._subgroup(sequence=left_grow, R=self._R, C=self._C, G=self._G)
                right_routines = super()._subgroup(sequence=right_grow, R=self._R, C=self._C, G=self._G)

                if routines_l_k.is_empty():
                    routines_l_k = self.__union_routines(left_routines, right_routines)
                else:
                    join_routines = self.__union_routines(left_routines, right_routines)
                    routines_l_k = self.__union_routines(routines_l_k, join_routines)

            if routines_l_k.is_empty():
                break

            self.__hierarchical_routines.add_routine(routines_l_k)
            actual_length += 1

        self.__already_fitted = True

    def get_results(self) -> HierarchyRoutine:
        """
        Returns the discovered routines as a `HierarchyRoutine` object.

        Returns:
            `HierarchyRoutine`. The discovered routines as a `HierarchyRoutine` object.

        Note:
            The `HierarchyRoutine` object provides methods and properties to further explore and manipulate the discovered routines.

        Examples:
            >>> import pandas as pd
            >>> time_series = pd.Series([1, 3, 6, 4, 2, 1, 2, 3, 6, 4, 1, 1, 3, 6, 4, 1])
            >>> time_series.index = pd.date_range(start="2024-01-01", periods=len(time_series))
            >>> drgs = DRGS(length_range=(3, 8), R=2, C=3, G=4, epsilon=0.5)
            >>> drgs.fit(time_series)
            >>> print(drgs.get_results())

        """
        return self.__hierarchical_routines

    def show_results(self) -> None:
        """
        Displays the discovered routines after fitting the model to the time series data.

        This method prints out detailed information about each discovered routine, including the centroid of each cluster, the subsequence instances forming the routine, and the dates/times these routines occur.

        Note:
            This method should be called after the `fit` method to ensure that routines have been discovered and are ready to be displayed.

        Examples:
            >>> import pandas as pd
            >>> time_series = pd.Series([1, 3, 6, 4, 2, 1, 2, 3, 6, 4, 1, 1, 3, 6, 4, 1])
            >>> time_series.index = pd.date_range(start="2024-01-01", periods=len(time_series))
            >>> drgs = DRGS(length_range=(3, 8), R=2, C=3, G=4, epsilon=0.5)
            >>> drgs.fit(time_series)
            >>> drgs.show_results()
        """

        if not self.__already_fitted:
            raise RuntimeError("The model has not been fitted yet. Please call the fit method before using this method")

        for length, routine in self.__hierarchical_routines.items:
            print(f"\n\t\t\t\t\t\t\t\t\t ROUTINES OF LENGTH {length}")
            print("_" * 100)
            for i, b in enumerate(routine):
                print(f"\tCentroid {i + 1}: {b.centroid}")
                print(f"\tRoutine {i + 1}: {b.get_sequences().get_subsequences()}")
                print(f"\tDate {i + 1}: {b.get_dates()}")
                print(f"\tStarting Points {i + 1}: ", b.get_starting_points())
                if len(routine) > 1 and i < len(routine) - 1:
                    print("\n\t", "-" * 80)

    def plot_hierarchical_results(self, title_fontsize: int = 20, show_xticks: bool = True,
                                  xticks_fontsize: int = 20, yticks_fontsize: int = 20,
                                  labels_fontsize: int = 20, figsize: tuple[int, int] = (30, 10),
                                  coloured_text_fontsize: int = 20, text_fontsize: int = 15,
                                  linewidth_bars: Union[int, float] = 1.5,
                                  xlim: Optional[tuple[int, int]] = None,
                                  save_dir: Optional[str] = None):

        """
        This method uses matplotlib to plot the results of the algorithm.
        The plot shows the time series data with vertical dashed lines indicating the start of each discovered routine.
        The color of each routine is determined by the order in which they were discovered.
        Each row in the plot represents a different hierarchy level,
        and each column represents a different routine within that hierarchy level.

        Parameters:
            * title_fontsize: `int` (default is 20). Size of the title plot.
            * show_xticks: `bool` (default is True). Whether to show the xticks or not.
            * xticks_fontsize: `int` (default is 20). Size of the xticks.
            * yticks_fontsize: `int (default is 20)`. Size of the yticks.
            * labels_fontsize: `int` (default is 20). Size of the labels.
            * figsize: `tuple[int, int]` (default is (30, 10)). Size of the figure.
            * coloured_text_fontsize: `int` (default is 20). Size of the coloured text.
            * text_fontsize: `int` (default is 15). Size of the text.
            * linewidth_bars: `int` | `float` (default is 1.5). Width of the bars in the plot.
            * xlim: `tuple[int, int]` (default is None). Limit of the x-axis with starting points.
            * save_dir: `str` (default is None). Directory to save the plot.

        Notes:
            This method has to be executed after the fit method to ensure that routines have been discovered and are ready to be displayed.

        Examples:
            >>> import pandas as pd
            >>> time_series = pd.Series([1, 3, 6, 4, 2, 1, 2, 3, 6, 4, 1, 1, 3, 6, 4, 1])
            >>> time_series.index = pd.date_range(start="2024-01-01", periods=len(time_series))
            >>> drgs = DRGS(length_range=(3, 8), R=2, C=3, G=4, epsilon=0.5)
            >>> drgs.fit(time_series)
            >>> drgs.plot_hierarchical_results()
        """
        # Check the validity of the parameters
        args = locals()
        super()._check_plot_params(**args)

        # Check if the model has been fitted before plotting
        if not self.__already_fitted:
            raise RuntimeError("The model has not been fitted yet. Please call the fit method before using this method")

        # Calculate the number of clusters per routine and determine plot dimensions
        n_cluster_per_routine = [len(routine) for routine in self.__hierarchical_routines.values]
        n_columns = max(n_cluster_per_routine)
        n_rows = len(self.__hierarchical_routines)
        maximum = max(self.time_series)

        # Set default x-axis limits if not provided
        xlim = xlim or (0, len(self.time_series))

        # Create the figure, grid layout and base colors for the plot
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(n_rows, n_columns, figure=fig)
        base_colors = cm.rainbow(np.linspace(0, 1, n_columns))

        # Plot each routine in the hierarchical routines
        for i, (length, routine) in enumerate(self.__hierarchical_routines.items):
            for j, cluster in enumerate(routine):
                # Add subplot for each cluster; use entire row if only one cluster
                fig.add_subplot(gs[i, :]) if len(routine) == 1 else fig.add_subplot(gs[i, j])

                # Set the colors for the time series bars
                colors = ["gray"] * len(self.time_series)
                for sp in cluster.get_starting_points():
                    # Plot a vertical-dashed line at the starting point of each subsequence in the cluster within the x-axis limits
                    if xlim[0] <= sp <= xlim[1]:
                        plt.axvline(x=sp, color=base_colors[j], linestyle="--")
                        # Annotate the time series points with their values
                        for k in range(cluster.length_cluster_subsequences):
                            if sp + k <= xlim[1]:
                                plt.text(x=sp + k - 0.05, y=self.time_series[sp + k] - 0.8,
                                         s=f"{self.time_series[sp + k]}", fontsize=coloured_text_fontsize,
                                         backgroundcolor="white", color=base_colors[j])

                                colors[sp + k] = base_colors[j]

                # Annotate the value of each bar that is not coloured
                for k in range(len(self.time_series)):
                    # if the bar is not coloured (its value is gray and not an array), annotate the value
                    if not isinstance(colors[k], np.ndarray):
                        plt.text(x=k - 0.05, y=self.time_series[k] + 0.8,
                                 s=f"{self.time_series[k]}", fontsize=text_fontsize,
                                 color="black")

                # Customize the title for each subplot
                plt.title(f"Hierarchy {length} Routine {j + 1}", fontsize=title_fontsize)

                # Plot the time series data as a bar plot
                plt.bar(np.arange(0, len(self.time_series)), self.time_series.values,
                        color=colors, edgecolor="black", linewidth=linewidth_bars)

                if show_xticks:
                    # Set the ticks on the x-axis and y-axis
                    plt.xticks(ticks=np.arange(xlim[0], xlim[1]),
                               labels=np.arange(xlim[0], xlim[1]),
                               fontsize=xticks_fontsize)
                else:
                    plt.xticks([])

                plt.yticks(fontsize=yticks_fontsize)

                # Set the labels for the x-axis and y-axis
                plt.xlabel("Starting Points", fontsize=labels_fontsize)
                plt.ylabel("Magnitude", fontsize=labels_fontsize)

                # Set the limits of the x-axis and y-axis
                plt.xlim(xlim[0] - 0.5, xlim[1])
                plt.ylim(0, maximum + 1)

        # Adjust the layout of the plot
        plt.tight_layout()

        # Save the plot to the specified directory if provided
        if save_dir:
            plt.savefig(save_dir)

        # Display the plot
        plt.show()


if __name__ == "__main__":
    time_series = pd.Series([1, 3, 6, 4, 2, 1, 2, 3, 6, 4, 1, 1, 3, 6, 4, 1])
    time_series = pd.Series(
        [1, 1, 1, 1, 5, 6, 9, 7, 5, 2, 1, 1, 1, 2, 5, 6, 9, 7, 4, 1, 1, 6, 6, 9, 7, 6, 1, 1, 1, 1, 1])
    # repeat the time series 10 times
    # time_series = pd.concat([time_series] * 10)
    time_series.index = pd.date_range(start="2024-01-01", periods=len(time_series))

    drgs = DRGS(length_range=(3, 13), R=2, C=3, G=4, epsilon=1, L=3)
    drgs.fit(time_series)
    drgs.plot_hierarchical_results(xticks_fontsize=10, coloured_text_fontsize=20, text_fontsize=15,
                                   show_xticks=False)
