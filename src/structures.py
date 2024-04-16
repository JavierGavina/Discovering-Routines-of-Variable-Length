"""
Data Structures.
-----------

This script defines the data structures needed for the algorithm of routine detection.

The module contains the following public classes:

Subsequence
-----------

Basic data structure.

    **Parameters**:
        * ``instance``: np.ndarray, the instance of the subsequence
        * ``date``: datetime.date, the date of the subsequence
        * ``starting_point``: int, the starting point of the subsequence

    **Public methods:**
        * ``get_instance() -> np.ndarray``: returns the instance of the subsequence
        * ``get_date() -> date``: returns the date of the subsequence
        * ``get_starting_point() -> int``: returns the starting point of the subsequence
        * ``to_collection() -> list[dict]``: returns the subsequence as a dictionary
        * ``magnitude() -> float``: returns the magnitude of the subsequence
        * ``inverse_magnitude() -> float``: returns the inverse magnitude of the subsequence
        * ``distance(other: Subsequence | np.ndarray) -> float``: returns the distance between the subsequence and another subsequence or array

    **Examples**:
        >>> subsequence = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
        >>> subsequence.get_instance()
        np.array([1, 2, 3, 4])

        >>> subsequence.get_date()
        datetime.date(2021, 1, 1)

        >>> subsequence.get_starting_point()
        0

        >>> subsequence.to_collection()
        {'instance': np.array([1, 2, 3, 4]), 'date': datetime.date(2021, 1, 1), 'starting_point': 0}

        >>> subsequence.magnitude()
        4

        >>> subsequence.distance(np.array([1, 2, 3, 4]))
        0

Sequence:
---------

Represents a sequence of subsequences

    **Parameters**:
        * ``subsequence``: Optional[Subsequence], the subsequence to add to the sequence. None is the default value

    **Properties**:
        *Getters:*
            * ``length_subsequences``: int, the length of the subsequences in the sequence

    **Public Methods:**
        * ``add_sequence(new: Subsequence)``: adds a Subsequence instance to the Sequence
        * ``get_by_starting_point(starting_point: int) -> Subsequence``: returns the subsequence with the specified starting point
        * ``set_by_starting_point(starting_point: int, new_sequence: Subsequence)``: sets the subsequence with the specified starting point
        * ``get_starting_points() -> list[int]``: returns the starting points of the subsequences
        * ``get_dates() -> list[dates]``: returns the dates of the subsequences
        * ``get_subsequences() -> list[np.ndarray]``: returns the instances of the subsequences
        * ``to_collection() -> list[dict]``: returns the sequence as a list of dictionaries

    **Examples**:

        >>> sequence = Sequence()
        >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
        >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
        >>> sequence.get_by_starting_point(0)
        Subsequence(instance=np.array([1, 2, 3, 4]), date=datetime.date(2021, 1, 1), starting_point=0)

        >>> sequence.get_starting_points()
        [0, 4]

        >>> sequence.get_dates()
        [datetime.date(2021, 1, 1), datetime.date(2021, 1, 2)]

        >>> sequence.get_subsequences()
        [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])]

        >>> sequence.to_collection()
        [{'instance': np.array([1, 2, 3, 4]), 'date': datetime.date(2021, 1, 1), 'starting_point': 0},
         {'instance': np.array([5, 6, 7, 8]), 'date': datetime.date(2021, 1, 2), 'starting_point': 4}]

        >>> sequence.length_subsequences
        4


Cluster
-------

Represents a cluster of subsequences

    **Parameters**:
        * ``centroid``: `np.ndarray`, the centroid of the cluster
        * ``instances``: `Sequence`, the sequences of subsequences

    **Properties**:
        *Getters:*
            * ``centroid: np.ndarray``: returns the centroid of the cluster
            * ``length_cluster_subsequences: int``: returns the length of the subsequences in the cluster
        *Setters:*
            * ``centroid: np.ndarray``: sets the centroid of the cluster

    **Public Methods:**
        * ``add_instance(new_subsequence: Subsequence)``: adds a subsequence to the cluster
        * ``get_sequences() -> Sequence``: returns the sequences of the cluster
        * ``update_centroid()``: updates the centroid of the cluster
        * ``get_starting_points() -> list[int]``: returns the starting points of the subsequences
        * ``get_dates() -> list[dates]``: returns the dates of the subsequences
        * ``cumulative_magnitude() -> float``: returns the cumulative magnitude of the cluster

    **Examples**:

        >>> subsequence1 = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
        >>> subsequence2 = Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)

        >>> sequence = Sequence(subsequence=subsequence1)
        >>> cluster = Cluster(np.array([1, 1, 1, 1]), sequence)

        >>> cluster.get_sequences()
        Sequence(
            length_subsequences=4,
            list_sequences=[
                Subsequence(
                    instance=np.array([1, 2, 3, 4]),
                    date=datetime.date(2021, 1, 1),
                    starting_point=0
                )
            ]

        >>> cluster.add_instance(subsequence2)
        >>> cluster.get_sequences()
        Sequence(
            length_subsequences=4,
            list_sequences=[
                Subsequence(
                    instance=np.array([1, 2, 3, 4]),
                    date=datetime.date(2021, 1, 1),
                    starting_point=0
                ),
                Subsequence(
                    instance=np.array([5, 6, 7, 8]),
                    date=datetime.date(2021, 1, 2),
                    starting_point=4
                )
            ]

        >>> cluster.update_centroid()
        >>> cluster.centroid
        np.array([3.0, 4.0, 5.0, 6.0])

        >>> cluster.get_starting_points()
        [0, 4]

        >>> cluster.get_dates()
        [datetime.date(2021, 1, 1), datetime.date(2021, 1, 2)]

        >>> cluster.cumulative_magnitude()
        12.0


Routines
--------

Represents a collection of clusters

    **Parameters**:
        * ``cluster: Optional[Cluster]``, the cluster to add to the collection. Default is None

    **Public Methods**:
        * ``add_routine(new_routine: Cluster)``: adds a cluster to the collection
        * ``drop_indexes(to_drop: list[int])``: drops the clusters with the specified indexes
        * ``get_routines() -> list[Cluster]``: returns the clusters from the `Routines`
        * ``get_centroids() -> list[np.ndarray]``: returns the centroids of the clusters
        * ``to_collection() -> list[dict]``: returns the routines as a list of dictionaries
        * ``is_empty() -> bool``: returns True if the collection is empty, False otherwise

    **Examples**:

        >>> subsequence1 = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
        >>> subsequence2 = Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)

        >>> sequence = Sequence(subsequence=subsequence1)
        >>> cluster1 = Cluster(np.array([1, 1, 1, 1]), sequence)

        >>> sequence = Sequence(subsequence=subsequence2)
        >>> cluster2 = Cluster(np.array([5, 5, 5, 5]), sequence)

        >>> routines = Routines(cluster=cluster1)
        >>> routines.add_routine(cluster2)

        >>> routines.get_routines()
        [Cluster(
            centroid=np.array([1, 1, 1, 1]),
            sequences=Sequence(
                length_subsequences=4,
                list_sequences=[
                    Subsequence(
                        instance=np.array([1, 2, 3, 4]),
                        date=datetime.date(2021, 1, 1),
                        starting_point=0
                    )
                ]
            )
        ), Cluster(
            centroid=np.array([5, 5, 5, 5]),
            sequences=Sequence(
                length_subsequences=4,
                list_sequences=[
                    Subsequence(
                        instance=np.array([5, 6, 7, 8]),
                        date=datetime.date(2021, 1, 2),
                        starting_point=4
                    )
                ]
            )
        )]

        >>> routines.get_centroids()
        [np.array([1, 1, 1, 1]), np.array([5, 5, 5, 5])]

        >>> routines.to_collection()
        [{'centroid': np.array([1, 1, 1, 1]), 'sequences': [{'instance': np.array([1, 2, 3, 4]), 'date': datetime.date(2021, 1, 1), 'starting_point': 0}], 'length_subsequences': 4},
         {'centroid': np.array([5, 5, 5, 5]), 'sequences': [{'instance': np.array([5, 6, 7, 8]), 'date': datetime.date(2021, 1, 2), 'starting_point': 4}], 'length_subsequences': 4}]

        >>> routines.is_empty()
        False

        >>> routines.drop_indexes([0])
        >>> routines.get_routines()
        [Cluster(
            centroid=np.array([5, 5, 5, 5]),
            sequences=Sequence(
                length_subsequences=4,
                list_sequences=[
                    Subsequence(
                        instance=np.array([5, 6, 7, 8]),
                        date=datetime.date(2021, 1, 2),
                        starting_point=4
                    )
                ]
            )
        )]

HierarchyRoutine
----------------

Represents a hierarchy of routines,
where the hierarchy corresponds to the length of the subsequences for each cluster in the routine.
For each hierarchy, exists one routine with the correspondent length of the subsequences.

    **Parameters**:
        * ``routine: Optional[Routines]``, the routine to add to the hierarchy. Default is None

    **Public Methods**:
        * ``add_routine(new_routine: Routines)``: adds a routine to the hierarchy
        * ``to_dictionary() -> dict``: returns the hierarchy as a dictionary

    **Properties**:
        *Getters:*
            * ``keys: list[int]``: returns the list with all the hierarchies registered
            * ``values: list[Routines]``: returns a list with all the routines
            * ``items: Iterator[tuple[int, Routines]]``: returns a iterator as a zip object with the hierarchy and the routine

    **Examples**:

            >>> subsequence1 = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> subsequence2 = Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)

            >>> sequence = Sequence(subsequence=subsequence1)
            >>> cluster1 = Cluster(np.array([1, 1, 1, 1]), sequence)

            >>> sequence = Sequence(subsequence=subsequence2)
            >>> cluster2 = Cluster(np.array([5, 5, 5, 5]), sequence)

            >>> routines = Routines(cluster=cluster1)
            >>> routines.add_routine(cluster2)

            >>> hierarchy = HierarchyRoutine(routine=routines)
            >>> hierarchy.to_dictionary()
            {4: [{'centroid': np.array([1, 1, 1, 1]), 'sequences': [{'instance': np.array([1, 2, 3, 4]), 'date': datetime.date(2021, 1, 1), 'starting_point': 0}]},
                 {'centroid': np.array([5, 5, 5, 5]), 'sequences': [{'instance': np.array([5, 6, 7, 8]), 'date': datetime.date(2021, 1, 2), 'starting_point': 4}]}]}

            >>> hierarchy.keys
            [4]

            >>> hierarchy.values
            [Routines(
                centroid=np.array([1, 1, 1, 1]),
                sequences=Sequence(
                    length_subsequences=4,
                    list_sequences=[
                        Subsequence(
                            instance=np.array([1, 2, 3, 4]),
                            date=datetime.date(2021, 1, 1),
                            starting_point=0
                        )
                    ]
                )
            ), Routines(
                centroid=np.array([5, 5, 5, 5]),
                sequences=Sequence(
                    length_subsequences=4,
                    list_sequences=[
                        Subsequence(
                            instance=np.array([5, 6, 7, 8]),
                            date=datetime.date(2021, 1, 2),
                            starting_point=4
                        )
                    ]
                )
            )]

            >>> for key, value in hierarchy.items:
            ...     print(key, value)
            4 Routines(
                centroid=np.array([1, 1, 1, 1]),
                sequences=Sequence(
                    length_subsequences=4,
                    list_sequences=[
                        Subsequence(
                            instance=np.array([1, 2, 3, 4]),
                            date=datetime.date(2021, 1, 1),
                            starting_point=0
                        )
                    ]
                )
            )
"""
import networkx as nx
import numpy as np
import datetime
from typing import Union, Optional, Iterator

import matplotlib.pyplot as plt
import re

import copy


class Subsequence:
    """
    Basic data structure to represent a subsequence from a sequence which belongs to time series

    Parameters:
    __________

        * ``instance: np.ndarray``, the instance of the subsequence
        * ``date: datetime.date``, the date of the subsequence
        * ``starting_point: int``, the starting point of the subsequence

    Public Methods:
    __________

        * ``get_instance() -> np.ndarray``: returns the instance of the subsequence
        * ``get_date() -> date``: returns the date of the subsequence
        * ``get_starting_point() -> int``: returns the starting point of the subsequence
        * ``to_collection() -> list[dict]``: returns the subsequence as a dictionary
        * ``magnitude() -> float``: returns the magnitude of the subsequence
        * ``distance(other: Subsequence | np.ndarray) -> float``: returns the distance between the subsequence and another subsequence or array

    Examples:

        >>> subsequence = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
        >>> subsequence.get_instance()
        np.array([1, 2, 3, 4])

        >>> subsequence.get_date()
        datetime.date(2021, 1, 1)

        >>> subsequence.get_starting_point()
        0

        >>> subsequence.to_collection()
        {'instance': np.array([1, 2, 3, 4]), 'date': datetime.date(2021, 1, 1), 'starting_point': 0}

        >>> subsequence.magnitude()
        4

        >>> subsequence.distance(np.array([1, 2, 3, 4]))
        0
    """

    def __init__(self, instance: np.ndarray, date: datetime.date, starting_point: int) -> None:
        """
        Parameters:
            * instance: `np.ndarray`, the instance of the subsequence
            * date: `datetime.date`, the date of the subsequence
            * starting_point: `int`, the starting point of the subsequence

        Raises:
            TypeError: if the parameters are not of the correct type

        Examples:
            >>> subsequence = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
        """

        self.__check_type(instance, date, starting_point)
        self.__instance = instance
        self.__date = date
        self.__starting_point = starting_point

    @staticmethod
    def __check_type(instance: np.ndarray, date: datetime.date, starting_point: int) -> None:
        """
        Checks the type of the parameters

        Parameters:
            * instance: `np.ndarray`, the instance of the subsequence
            * date: `datetime.date`, the date of the subsequence
            * starting_point: `int`, the starting point of the subsequence

        Raises:
            TypeError: if the parameters are not of the correct type
        """

        # Check if the instance is an array
        if not isinstance(instance, np.ndarray):
            raise TypeError(f"Instances must be an arrays. Got {type(instance).__name__} instead")

        # Check if the date is a datetime.date
        if not isinstance(date, datetime.date):
            raise TypeError(f"Date must be a timestamps. Got {type(date).__name__} instead")

        # Check if the starting point is an integer
        if not isinstance(starting_point, int):
            raise TypeError(f"starting_point must be a integer. Got {type(starting_point).__name__} instead")

    def __repr__(self):
        """
        Returns the string representation of the subsequence

        Returns:
            str. The string representation of the subsequence

        Examples:
            >>> subsequence = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> print(subsequence)
            Subsequence(
                instance=np.array([1, 2, 3, 4]),
                date=datetime.date(2021, 1, 1),
                starting_point=0
            )
        """

        return f"Subsequence(\n\t instance={self.__instance} \n\t date={self.__date} \n\t starting point = {self.__starting_point}\n)"

    def __str__(self):
        """
        Returns the string representation of the subsequence

        Returns:
            str. The string representation of the subsequence

        Examples:
            >>> subsequence = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> print(subsequence)
            Subsequence(
                instance=np.array([1, 2, 3, 4]),
                date=datetime.date(2021, 1, 1),
                starting_point=0
            )
        """

        return f"Subsequence(\n\t instances={self.__instance} \n\t date={self.__date} \n\t starting point = {self.__starting_point}\n)"

    def __len__(self) -> int:
        """
        Returns the length of the subsequence

        Returns:
            `int`. The length of the subsequence

        Examples:
            >>> subsequence = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> len(subsequence)
            4
        """
        return len(self.__instance)

    def __getitem__(self, index: int) -> int | float:
        """
        Get the item at the specified index in the subsequence

        Parameters:
            * index: `int`. The index of the item

        Returns:
            `float` | `int`. The item at the specified index in the subsequence

        Raises:
            TypeError: if the parameter is not of the correct type
            IndexError: if the index is out of range

        Examples:
            >>> subsequence = Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 1), 0)
            >>> subsequence[2]
            7
        """

        # Check if the index is an integer
        if not isinstance(index, int):
            raise TypeError(f"index must be an integer. Got {type(index).__name__} instead")

        # Check if the index is within the range of the list
        if not 0 <= index < len(self.__instance):
            raise IndexError(f"index {index} out of range (0, {len(self.__instance) - 1})")

        # If the item is a numpy integer or float, convert it to a Python integer or float and return it
        if isinstance(self.__instance[index], np.int32):
            return int(self.__instance[index])

        return float(self.__instance[index])

    def __eq__(self, other: 'Subsequence') -> bool:
        """
        Check if the subsequence is equal to another subsequence

        Parameters:
            other: `Subsequence`. The subsequence to compare

        Returns:
            `bool`. True if the subsequences are equal, False otherwise

        Raises:
            TypeError: if the parameter is not of the correct type

        Examples:
            >>> subsequence1 = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> subsequence2 = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> subsequence1 == subsequence2
            True

            >>> subsequence1 = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> subsequence2 = Subsequence(np.array([1, 2, 3, 4, 5]), datetime.date(2021, 1, 2), 2)
            >>> subsequence1 == subsequence2
            False
        """

        # Check if the parameter is an instance of Subsequence
        if not isinstance(other, Subsequence):
            raise TypeError(f"other must be an instance of Subsequence. Got {type(other).__name__} instead")

        # Check if they have the same length
        if len(self) != len(other):
            return False

        # Check if the instance, date, and starting point are equal
        if not np.array_equal(self.__instance, other.get_instance()):
            return False

        # Check if the date and starting point are equal
        if self.__date != other.get_date() or self.__starting_point != other.get_starting_point():
            return False

        return True

    def get_instance(self) -> np.ndarray:
        """
        Returns the instance of the subsequence

        Returns:
             `np.ndarray`. The instance of the `Subsequence`

        Examples:
            >>> subsequence = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> subsequence.get_instance()
            np.array([1, 2, 3, 4])
        """

        return self.__instance

    def get_date(self) -> datetime.date:
        """
        Returns the date of the subsequence

        Returns:
            `datetime.date`. The date of the `Subsequence`

        Examples:
            >>> subsequence = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> subsequence.get_date()
            datetime.date(2021, 1, 1)
        """

        return self.__date

    def get_starting_point(self) -> int:
        """
        Returns the starting point of the subsequence

        Returns:
             `int`. The starting point of the `Subsequence`

        Examples:
            >>> subsequence = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> subsequence.get_starting_point()
            0
        """

        return self.__starting_point

    def to_collection(self) -> dict:
        """
        Returns the subsequence as a dictionary

        Returns:
             `dict`. The subsequence as a dictionary

        Examples:
            >>> subsequence = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> subsequence.to_collection()
            {'instance': np.array([1, 2, 3, 4]), 'date': datetime.date(2021, 1, 1), 'starting_point': 0}
        """

        return {"instance": self.__instance, "date": self.__date, "starting_point": self.__starting_point}

    def magnitude(self) -> float:
        """
        Returns the magnitude of the subsequence as the maximum value

        Returns:
             `float`. The magnitude of the subsequence

        Examples:
            >>> subsequence = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> subsequence.magnitude()
            4.0
        """

        return np.max(self.__instance)

    def inverse_magnitude(self) -> float:
        """
        Returns the inverse magnitude of the subsequence as the minimum value

        Returns:
            `float`. The inverse magnitude of the subsequence

        Examples:
            >>> subsequence = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> subsequence.inverse_magnitude()
            1.0
        """

        return np.min(self.__instance)

    def distance(self, other: Union['Subsequence', np.ndarray]) -> float:
        """
        Returns the maximum absolute distance between the subsequence and another subsequence or array

        Parameters:
            * other: `Union[Subsequence, np.ndarray]`, the subsequence or array to compare

        Returns:
            `float`. The distance between the subsequence and another subsequence or array

        Raises:
            TypeError: if the parameter is not of the correct type
            ValueError: if the instances have different lengths

        Examples:
            >>> subsequence = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> subsequence.distance(np.array([1, 2, 3, 4]))
            0.0

            >>> subsequence.distance(Subsequence(np.array([1, 2, 3, 6]), datetime.date(2021, 1, 2), 2))
            2.0
        """

        # Check if the parameter is an instance of Subsequence or np.ndarray
        if isinstance(other, Subsequence):
            new_instance = other.get_instance()

        elif isinstance(other, np.ndarray):
            new_instance = other

        # If the parameter is not an instance of Subsequence or np.ndarray, raise an error
        else:
            raise TypeError(
                f"other must be an instance of Subsequence or np.ndarray. Got {type(other).__name__} instead")

        # Check if the instances have the same length
        if len(self.__instance) != len(new_instance):
            raise ValueError(
                f"The instances must have the same length. len(self)={len(self.__instance)} and len(other)={len(new_instance)}")

        return np.max(np.abs(self.__instance - new_instance))


class Sequence:
    """
    Represents a sequence of subsequences

    Parameters:
    _________
        * ``subsequence: Optional[Subsequence]``, the subsequence to add to the sequence. Default is `None`

    Properties:
    _________

    **Getters**
        * ``length_subsequences: int``. The length of the subsequences in the sequence

    Public Methods:
    _________

        * ``add_sequence(new: Subsequence)`` : adds a `Subsequence` instance to the `Sequence`
        * ``get_by_starting_point(starting_point: int)`` -> Subsequence: returns the subsequence with the specified starting point
        * ``set_by_starting_point(starting_point: int, new_sequence: Subsequence):`` sets the subsequence with the specified starting point
        * ``get_starting_points() -> list[int]:`` returns the starting points of the subsequences
        * ``get_dates() -> list[dates]:`` returns the dates of the subsequences
        * ``get_subsequences() -> list[np.ndarray]:`` returns the instances of the subsequences
        * ``to_collection() -> list[dict]:`` returns the sequence as a list of dictionaries

    Examples:
        >>> sequence = Sequence()
        >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
        >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
        >>> sequence.get_by_starting_point(0)
        Subsequence(instance=np.array([1, 2, 3, 4]), date=datetime.date(2021, 1, 1), starting_point=0)
        >>> sequence.get_starting_points()
        [0, 4]
        >>> sequence.get_dates()
        [datetime.date(2021, 1, 1), datetime.date(2021, 1, 2)]
        >>> sequence.get_subsequences()
        [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])]
        >>> sequence.to_collection()
        [{'instance': np.array([1, 2, 3, 4]), 'date': datetime.date(2021, 1, 1), 'starting_point': 0},
         {'instance': np.array([5, 6, 7, 8]), 'date': datetime.date(2021, 1, 2), 'starting_point': 4}]
    """

    def __init__(self, subsequence: Optional[Subsequence] = None) -> None:
        """
        Parameters:
            * subsequence: Optional[Subsequence], the subsequence to add to the sequence

        Raises:
            TypeError: if the parameter is not of the correct type

        Examples:
            >>> sequence1 = Sequence(subsequence=Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence2 = Sequence()
            >>> sequence2.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
        """

        # Initialize the length of the sequence
        self.__length = None

        # Check if the subsequence is a Subsequence instance
        if subsequence is not None:
            self.__check_validity_params(subsequence)

            # Make a deep copy of the subsequence
            new_subsequence = copy.deepcopy(subsequence)
            self.__list_sequences: list[Subsequence] = [new_subsequence]

            # Set the length of the sequence
            self.__length: int = len(subsequence)

        # If the subsequence is None, initialize an empty list
        else:
            self.__list_sequences: list[Subsequence] = []

    def __repr__(self):
        """
        Returns the string representation of the sequence

        Returns:
            str. The string representation of the sequence

        Examples:
            >>> sequence = Sequence(subsequence=Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> print(sequence)
            Sequence(
                length_subsequences=4,
                list_sequences=[
                    Subsequence(
                        instance=np.array([1, 2, 3, 4]),
                        date=datetime.date(2021, 1, 1),
                        starting_point=0
                    )
                ]
            )
        """

        out_string = "Sequence(\n\tlength_subsequences=" + f"{self.__length}" + "\n\tlist_sequences=[\n"
        for seq in self.__list_sequences:
            out_string += f" {seq},\n"

        out_string = out_string[:-2] + out_string[-1] + "]"
        return out_string

    def __str__(self):
        """
        Returns the string representation of the sequence

        Returns:
            `str`. The string representation of the sequence

        Examples:
            >>> sequence = Sequence(length=4, subsequence=Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> print(sequence)
            Sequence(
                length_subsequences=4,
                list_sequences=[
                    Subsequence(
                        instance=np.array([1, 2, 3, 4]),
                        date=datetime.date(2021, 1, 1),
                        starting_point=0
                    )
                ]
            )
        """

        out_string = "Sequence(\n\tlength_subsequences=" + f"{self.__length}" + "\n\tlist_sequences=[\n"
        for seq in self.__list_sequences:
            out_string += f" {seq},\n"

        out_string = out_string[:-2] + out_string[-1] + "]"
        return out_string

    def __len__(self) -> int:
        """
        Returns the number of `Subsequence` instances in the `Sequence`

        Returns:
            `int`. The number of subsequences in the sequence

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> len(sequence)
            2
        """

        return len(self.__list_sequences)

    def __getitem__(self, index: int) -> 'Subsequence':
        """
        Get the subsequence at the specified index in the sequence

        Parameters:
            * index: `int`. The index of the subsequence

        Returns:
            `Subsequence`. The subsequence at the specified index in the sequence

        Raises:
            TypeError: if the parameter is not of the correct type
            IndexError: if the index is out of range

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence[0]
            Subsequence(instance=np.array([1, 2, 3, 4]), date=datetime.date(2021, 1, 1), starting_point=0)
        """
        # Check if the index is an integer
        if not isinstance(index, int):
            raise TypeError(f"index must be an integer. Got {type(index).__name__} instead")

        # Check if the index is within the range of the list
        if not 0 <= index < len(self.__list_sequences):
            raise IndexError(f"index {index} out of range (0, {len(self.__list_sequences) - 1})")

        return self.__list_sequences[index]

    def __setitem__(self, index: int, value: 'Subsequence') -> None:
        """
        Set the value of the subsequence at the specified index in the sequence

        Parameters:
            * index: int. The index of the subsequence
            * value: Subsequence. The new subsequence

        Raises:
            TypeError: if the parameter is not of the correct type
            IndexError: if the index is out of range

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence[0] = Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)
            >>> sequence[0]
            Subsequence(instance=np.array([5, 6, 7, 8]), date=datetime.date(2021, 1, 2), starting_point=4)
        """

        # Check if the new_sequence is a Subsequence instance
        if not isinstance(value, Subsequence):
            raise TypeError(f"new_sequence must be an instance of Subsequence. Got {type(value).__name__} instead")

        # Check if the index is an integer
        if not isinstance(index, int):
            raise TypeError(f"index must be an integer. Got {type(index).__name__} instead")

        # Check if the index is within the range of the list
        if not 0 <= index < len(self.__list_sequences):
            raise IndexError(f"index {index} out of range (0, {len(self.__list_sequences) - 1})")

        self.__list_sequences[index] = value

    def __iter__(self):
        """
        Returns an iterator for each subsequence in the sequence

        Returns:
            iter. An iterator for each subsequence in the sequence

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> for subsequence in sequence:
            ...     print(subsequence)
            Subsequence(instance=np.array([1, 2, 3, 4]), date=datetime.date(2021, 1, 1), starting_point=0)
            Subsequence(instance=np.array([5, 6, 7, 8]), date=datetime.date(2021, 1, 2), starting_point=4)
        """

        return iter(self.__list_sequences)

    def __contains__(self, item: 'Subsequence') -> bool:
        """
        Check if the subsequence exists in the sequence

        Parameters:
            * item: `Subsequence`. The subsequence to check

        Returns:
            `bool`. `True` if the subsequence exists, `False` otherwise

        Raises:
            TypeError: if the parameter is not an instance of Subsequence

        Examples:
            >>> sequence = Sequence(subsequence=Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 1))
            >>> Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0) in sequence
            True

        """
        # Check if the new_sequence is a Subsequence instance
        if not isinstance(item, Subsequence):
            raise TypeError(f"new_sequence must be an instance of Subsequence. Got {type(item).__name__} instead")

        return item in self.__list_sequences

    def __delitem__(self, index: int) -> None:
        """
        Deletes the subsequence at the specified index in the sequence

        Parameters:
            * index: `int`. The index of the subsequence to delete

        Raises:
            TypeError: if the parameter is not of the correct type
            IndexError: if the index is out of range
        """
        if not isinstance(index, int):
            raise TypeError(f"index must be an integer. Got {type(index).__name__} instead")

        if not 0 <= index < len(self.__list_sequences):
            raise IndexError(f"index {index} out of range (0, {len(self.__list_sequences) - 1})")

        del self.__list_sequences[index]

    def __add__(self, other: 'Sequence') -> 'Sequence':
        """
        Concatenates two sequences together with the operator +

        Parameters:
            * other: `Sequence`. The sequence to concatenate

        Returns:
            `Sequence`. The concatenated sequence

        Raises:
            TypeError: if the parameter is not an instance of `Sequence`

        Examples:
            >>> sequence1 = Sequence()
            >>> sequence1.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence2 = Sequence()
            >>> sequence2.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 1))
            >>> new_sequence = sequence1 + sequence2
            >>> print(new_sequence)
            Sequence(
                list_sequences=[
                    Subsequence(
                        instance=np.array([1, 2, 3, 4]),
                        date=datetime.date(2021, 1, 1),
                        starting_point=0
                    ),
                    Subsequence(
                        instance=np.array([5, 6, 7, 8]),
                        date=datetime.date(2021, 1, 2),
                        starting_point=1
                    )
                ]
            )
        """
        if not isinstance(other, Sequence):
            raise TypeError(f"other must be an instance of Sequence. Got {type(other).__name__} instead")

        new_sequence = Sequence()
        new_sequence.__list_sequences = self.__list_sequences + other.__list_sequences
        return new_sequence

    def __eq__(self, other: 'Sequence') -> bool:
        """
        Check if the sequence is equal to another sequence

        Parameters:
            * other: `Sequence`. The sequence to compare

        Returns:
            `bool`. True if the sequences are equal, False otherwise

        Raises:
            TypeError: if the parameter is not of the correct type

        Examples:
            >>> sequence1 = Sequence()
            >>> sequence1.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence2 = Sequence()
            >>> sequence2.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence1 == sequence2
            True
        """

        # Check if the parameter is an instance of Sequence
        if not isinstance(other, Sequence):
            raise TypeError(f"other must be an instance of Sequence. Got {type(other).__name__} instead")

        # Check if the subsequences are equal
        return np.array_equal(self.get_subsequences(), other.get_subsequences())

    @property
    def length_subsequences(self) -> int:
        """
        Getter that returns the length of the subsequences in the sequence

        Returns:
            `int`. The length of the subsequences in the sequence

        Examples:
            >>> sequence = Sequence(subsequence=Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.length_subsequences
            4
        """

        return self.__length

    def __check_validity_params(self, subsequence: Subsequence) -> None:
        """
        Check if the parameters are valid

        Parameters:
            * subsequence: `Subsequence`. The subsequence to add to the sequence

        Raises:
            TypeError: if the parameters are not of the correct type
            ValueError: if the length of the subsequence is not the same as the length of the sequence
        """

        # Check if the subsequence is a Subsequence instance
        if not isinstance(subsequence, Subsequence):
            raise TypeError(f"subsequence must be an instance of Subsequence. Got {type(subsequence).__name__} instead")

        # Check if the length of the subsequence is the same as the length of the sequence
        if self.__length is not None and len(subsequence) != self.__length:
            raise ValueError(
                f"The length of the subsequence must be the same as the length of the Sequence. Got {len(subsequence)} instead of {self.__length}")

    def _already_exists(self, subsequence: 'Subsequence') -> bool:
        """
        Check if the subsequence already exists in the sequence

        Parameters:
            * subsequence: `Subsequence`. The subsequence to check

        Returns:
            `bool`. True if the `subsequence` already exists, `False` otherwise

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence._already_exists(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            True
        """

        self_collection = self.to_collection()
        new_self_collection = []

        # Is necessary to convert the arrays to list for checking properly if the new sequence exists
        for idx, dictionary in enumerate(self_collection):
            dictionary["instance"] = dictionary["instance"].tolist()
            new_self_collection.append(dictionary)

        # convert to collection and transform from array to list
        collection = subsequence.to_collection()
        collection = {"instance": collection["instance"].tolist()}

        return collection in new_self_collection

    def add_sequence(self, new: 'Subsequence') -> None:
        """
        Adds a subsequence to the sequence

        Parameters:
            * new: `Subsequence`. The subsequence to add

        Raises:
            TypeError: if the parameter is not of the correct type
            RuntimeError: if the subsequence already exists
            ValueError: if the length of the subsequence is not the same as the length of the sequence

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> print(sequence)
            Sequence(
                length_subsequences=4,
                list_sequences=[
                    Subsequence(
                        instance=np.array([1, 2, 3, 4]),
                        date=datetime.date(2021, 1, 1),
                        starting_point=0
                    ),
                    Subsequence(
                        instance=np.array([5, 6, 7, 8]),
                        date=datetime.date(2021, 1, 2),
                        starting_point=4
                    )
                ]
            )
        """
        # Check if the new sequence is a Subsequence instance
        if not isinstance(new, Subsequence):
            raise TypeError(f"new has to be an instance of Subsequence. Got {type(new).__name__} instead")

        # Check if the new sequence already exists
        if self._already_exists(new):
            raise RuntimeError("new sequence already exists ")

        # Check if the length of the subsequence is the same as the length of the sequence
        if self.__length is not None and len(new) != self.__length:
            raise ValueError(
                f"The length of the subsequence must be the same as the length of the Sequence. Got {len(new)} instead of {self.__length}")

        # If the sequence is empty, set the length of the sequence
        if len(self.__list_sequences) == 0:
            self.__length = len(new)

        self.__list_sequences.append(new)

    def get_by_starting_point(self, starting_point: int) -> Optional['Subsequence']:
        """
        Returns the subsequence with the specified starting point

        Parameters:
            * starting_point: `int`. The starting point of the subsequence

        Returns:
            Optional[Subsequence]. The subsequence with the specified starting point if it exists. Otherwise, None

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 1))
            >>> sequence.get_by_starting_point(0)
            Subsequence(instance=np.array([1, 2, 3, 4]), date=datetime.date(2021, 1, 1), starting_point=0)

            >>> sequence.get_by_starting_point(2)
            None
        """

        for subseq in self.__list_sequences:
            if subseq.get_starting_point() == starting_point:
                return subseq

        return None

    def set_by_starting_point(self, starting_point: int, new_sequence: 'Subsequence') -> None:
        """
        Sets the subsequence with the specified starting point

        Parameters:
            * starting_point: int. The starting point of the subsequence
            * new_sequence: Subsequence. The new subsequence

        Raises:
            TypeError: if the parameter is not of the correct type
            ValueError: if the starting point does not exist

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 1))
            >>> sequence.set_by_starting_point(0, Subsequence(np.array([9, 10, 11, 12]), datetime.date(2021, 1, 3), 0))
            >>> sequence.get_by_starting_point(0)
            Subsequence(instance=np.array([9, 10, 11, 12]), date=datetime.date(2021, 1, 3), starting_point=0)
        """

        # Check if the new_sequence is a Subsequence instance
        if not isinstance(new_sequence, Subsequence):
            raise TypeError(
                f"new_sequence must be an instance of Subsequence. Got {type(new_sequence).__name__} instead")

        # Iterate through the list to find the subsequence with the matching starting point
        for i, subseq in enumerate(self.__list_sequences):
            if subseq.get_starting_point() == starting_point:
                # Replace the found subsequence with the new one
                self.__list_sequences[i] = new_sequence
                return

        # If not found, raise an error indicating the starting point does not exist
        raise ValueError(
            f"The starting point {starting_point} doesn't exist. The available starting points are {self.get_starting_points()}")

    def get_starting_points(self, to_array: bool = False) -> list[int]:
        """
        Returns the starting points of the subsequences

        Parameters:
            * to_array: `bool`. If True, returns the starting points as a numpy array. Default is `False`

        Returns:
             `list[int]`. The starting points of the subsequences

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> sequence.get_starting_points()
            [0, 4]
        """

        sequence_starting_points = [subseq.get_starting_point() for subseq in self.__list_sequences]

        if to_array:
            return np.array(sequence_starting_points)

        return sequence_starting_points

    def get_dates(self) -> list[datetime.date]:
        """
        Returns the dates of the subsequences

        Returns:
             `list[datetime.date]`. The dates of the subsequences

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 1))
            >>> sequence.get_dates()
            [datetime.date(2021, 1, 1), datetime.date(2021, 1, 2)]
        """

        return [subseq.get_date() for subseq in self.__list_sequences]

    def get_subsequences(self, to_array: bool = False) -> Union[list[np.ndarray], np.ndarray]:
        """
        Returns the instances of the subsequences

        Returns:
             `list[np.ndarray]`. The instances of the subsequences

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 1))
            >>> sequence.get_subsequences()
            [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])]

            >>> sequence.get_subsequences(to_array=True)
            array([np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8]])
        """
        subsequences = [subseq.get_instance() for subseq in self.__list_sequences]

        if to_array:
            return np.array(subsequences)

        return subsequences

    def extract_components(self, flatten: bool = False) -> tuple[np.ndarray, list[datetime.date], list[int]]:
        """
        Extract the components of a sequence.

        Returns:
            `tuple[np.ndarray, list[datetime.date], list[int]]`. The subsequence, dates, and starting points of the sequence.

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3]), datetime.date(2024, 1, 1), 0))
            >>> subsequence, dates, starting_points = sequence.extract_components()
            >>> print(subsequence)
            [1 2 3]

            >>> print(dates)
            [datetime.date(2024, 1, 1)]

            >>> print(starting_points)
            [0]
        """

        subsequence = self.get_subsequences(to_array=True)

        if flatten:
            subsequence = subsequence.flatten()

        dates = self.get_dates()
        starting_points = self.get_starting_points()

        return subsequence, dates, starting_points

    def to_collection(self) -> list[dict]:
        """
        Returns the sequence as a list of dictionaries

        Returns:
             `list[dict]`. The sequence as a list of dictionaries

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 1))
            >>> sequence.to_collection()
            [{'instance': np.array([1, 2, 3, 4]), 'date': datetime.date(2021, 1, 1), 'starting_point': 0},
             {'instance': np.array([5, 6, 7, 8]), 'date': datetime.date(2021, 1, 2), 'starting_point': 1}]
        """

        collection = []
        for subseq in self.__list_sequences:
            collection.append({
                'instance': subseq.get_instance(),
                'date': subseq.get_date(),
                'starting_point': subseq.get_starting_point()
            })

        return collection


class Cluster:
    """
    Represents a cluster of subsequences from a sequence and a centroid.

    Parameters:
        * ``centroid: np.ndarray``, the centroid of the cluster
        * ``instances: Sequence``, the sequence of subsequences

    Properties:
    ________
        **Getters**:
            * ``centroid: np.ndarray``, the centroid of the cluster
            * ``length_cluster_subsequences: int``, the length of each subsequence in the cluster

        **Setters**:
            * ``centroid: np.ndarray | Subsequence``, the centroid of the cluster


    Public Methods:
    ________

        * ``add_instance(new_subsequence: Subsequence)``: adds a subsequence to the cluster
        * ``update_centroid()``: updates the centroid of the cluster
        * ``get_sequences() -> Sequence``: returns the sequences of the cluster
        * ``get_starting_points() -> list[int]``: returns the starting points of the subsequences
        * ``get_dates() -> list[date]``: returns the dates of the subsequences
        * ``cumulative_magnitude() -> float``: returns the cumulative magnitude of the cluster


    Examples:

        >>> sequence = Sequence()
        >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
        >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))

        >>> cluster = Cluster(np.array([1, 1, 1, 1]), sequence)
        >>> cluster.get_sequences().to_collection()
        [{'instance': np.array([1, 2, 3, 4]), 'date': datetime.date(2021, 1, 1), 'starting_point': 0},
         {'instance': np.array([5, 6, 7, 8]), 'date': datetime.date(2021, 1, 2), 'starting_point': 4}]

        >>> cluster.get_starting_points()
        [0, 4]

        >>> cluster.get_dates()
        [datetime.date(2021, 1, 1), datetime.date(2021, 1, 2)]

        >>> cluster.centroid
        np.array([1, 1, 1, 1])

        >>> cluster.centroid = np.array([1, 2, 3, 4])
        >>> cluster.centroid
        np.array([1, 2, 3, 4])

        >>> cluster.update_centroid()
        >>> cluster.centroid
        np.array([3, 4, 5, 6])
    """

    def __init__(self, centroid: np.ndarray, instances: 'Sequence') -> None:
        """
        Parameters:
            * centroid: `np.ndarray`, the centroid of the cluster
            * instances: `Sequence`, the sequence of subsequences

        Raises:
            TypeError: if the centroid is not an instance of `np.ndarray` or the instances are not an instance of `Sequence`

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> cluster = Cluster(4, np.array([3, 4, 5, 6]), sequence)
        """

        # Check the validity of the parameters
        self.__check_validity_params(centroid, instances)

        # Make a deep copy of the instances to avoid modifying the original sequence
        new_instances = copy.deepcopy(instances)

        # Set the length centroid and the instances
        self.__length: int = new_instances.length_subsequences
        self.__centroid: np.ndarray = centroid
        self.__instances: Sequence = new_instances

    def __str__(self):
        """
        Returns the string representation of the cluster

        Returns:
            `str`. The string representation of the cluster

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> cluster = Cluster(np.array([3, 4, 5, 6]), sequence)
            >>> print(cluster)
            Cluster(
                -Centroid: np.array([3, 4, 5, 6])
                -Instances: [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])]
                -Dates: [datetime.date(2021, 1, 1), datetime.date(2021, 1, 2)]
                -Starting Points: [0, 4]
            )
        """

        out_string = f"Cluster(\n\t -Centroid: {self.__centroid} \n"
        out_string += f"\t -Instances: {[instance for instance in self.__instances.get_subsequences()]}\n"
        out_string += f"\t -Dates: {[date for date in self.__instances.get_dates()]}\n"
        out_string += f"\t -Starting Points: {[sp for sp in self.__instances.get_starting_points()]}\n)"
        return out_string

    def __repr__(self):
        """
        Returns the string representation of the cluster

        Returns:
            `str`. The string representation of the cluster

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> cluster = Cluster(np.array([3, 4, 5, 6]), sequence)
            >>> print(cluster)
            Cluster(
                -Centroid: np.array([3, 4, 5, 6])
                -Instances: [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])]
                -Dates: [datetime.date(2021, 1, 1), datetime.date(2021, 1, 2)]
                -Starting Points: [0, 4]
            )
        """

        out_string = f"Cluster(\n\t -Centroid: {self.__centroid} \n"
        out_string += f"\t -Instances: {[instance for instance in self.__instances.get_subsequences()]}\n"
        out_string += f"\t -Dates: {[date for date in self.__instances.get_dates()]}\n"
        out_string += f"\t -Starting Points: {[sp for sp in self.__instances.get_starting_points()]}\n)"
        return out_string

    def __len__(self) -> int:
        """
        Returns the number of instances in the cluster

        Returns:
            `int`. The number of instances in the cluster

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> cluster = Cluster(np.array([3, 4, 5, 6]), sequence)
            >>> len(cluster)
            2
        """

        return len(self.__instances)

    def __getitem__(self, index: int) -> 'Subsequence':
        """
        Get the subsequence at the specified index in the cluster

        Parameters:
            * index: `int`. The index of the subsequence

        Returns:
            `Subsequence`. The subsequence at the specified index in the cluster

        Raises:
            TypeError: if the parameter is not of the correct type
            IndexError: if the index is out of range

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> cluster = Cluster(np.array([3, 4, 5, 6]), sequence)
            >>> cluster[0]
            Subsequence(instance=np.array([1, 2, 3, 4]), date=datetime.date(2021, 1, 1), starting_point=0)
        """

        # Check if the index is an integer
        if not isinstance(index, int):
            raise TypeError(f"index must be an integer. Got {type(index).__name__} instead")

        # Check if the index is within the range of the list
        if not 0 <= index < len(self.__instances):
            raise IndexError(f"index {index} out of range (0, {len(self.__instances) - 1})")

        return self.__instances[index]

    def __setitem__(self, index: int, value: 'Subsequence') -> None:
        """
        Set the value of the subsequence at the specified index in the cluster

        Parameters:
            * index: `int`. The index of the subsequence
            * value: `Subsequence`. The new subsequence

        Raises:
            TypeError: if the parameter is not of the correct type
            IndexError: if the index is out of range

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> cluster = Cluster(np.array([3, 4, 5, 6]), sequence)
            >>> cluster[0] = Subsequence(np.array([9, 10, 11, 12]), datetime.date(2021, 1, 3), 0)
            >>> cluster[0]
            Subsequence(instance=np.array([9, 10, 11, 12]), date=datetime.date(2021, 1, 3), starting_point=0)
        """

        # Check if the new_sequence is a Subsequence instance
        if not isinstance(value, Subsequence):
            raise TypeError(f"new_sequence must be an instance of Subsequence. Got {type(value).__name__} instead")

        # Check if the index is an integer
        if not isinstance(index, int):
            raise TypeError(f"index must be an integer. Got {type(index).__name__} instead")

        # Check if the index is within the range of the list
        if not 0 <= index < len(self.__instances):
            raise IndexError(f"index {index} out of range (0, {len(self.__instances) - 1})")

        self.__instances[index] = value

    def __iter__(self) -> iter:
        """
        Returns an iterator for each subsequence in the cluster's instances

        Returns:
            `iter`. An iterator for each subsequence in the cluster's instances

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> cluster = Cluster(np.array([3, 4, 5, 6]), sequence)
            >>> for subsequence in cluster:
            ...     print(subsequence)
            Subsequence(instance=np.array([1, 2, 3, 4]), date=datetime.date(2021, 1, 1), starting_point=0)
            Subsequence(instance=np.array([5, 6, 7, 8]), date=datetime.date(2021, 1, 2), starting_point=4)
        """

        return iter(self.__instances)

    def __contains__(self, item: 'Subsequence') -> bool:
        """
        Check if the subsequence exists in the cluster's instances

        Parameters:
            * item: `Subsequence`. The subsequence to check

        Returns:
            `bool`. `True` if the subsequence exists, `False` otherwise

        Raises:
            TypeError: if the parameter is not an instance of `Subsequence`

        Examples:
            >>> sequence = Sequence(subsequence=Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> cluster = Cluster(np.array([3, 4, 5, 6]), sequence)
            >>> item = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> item in cluster
            True
        """

        # Check if the new_sequence is a Subsequence instance
        if not isinstance(item, Subsequence):
            raise TypeError(f"new_sequence must be an instance of Subsequence. Got {type(item).__name__} instead")

        return item in self.__instances

    def __delitem__(self, index: int) -> None:
        """
        Deletes the subsequence at the specified index in the cluster's instances

        Parameters:
            * index: `int`. The index of the subsequence to delete

        Raises:
            TypeError: if the parameter is not of the correct type
            IndexError: if the index is out of range

        Examples:
            >>> sequence = Sequence(subsequence=Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> cluster = Cluster(np.array([3, 4, 5, 6]), sequence)
            >>> del cluster[0]
            >>> print(cluster)
            Cluster(
                -Centroid: np.array([3, 4, 5, 6])
                -Instances: [np.array([5, 6, 7, 8])]
                -Dates: [datetime.date(2021, 1, 2)]
                -Starting Points: [4]
            )
        """

        if not isinstance(index, int):
            raise TypeError(f"index must be an integer. Got {type(index).__name__} instead")

        if not 0 <= index < len(self.__instances):
            raise IndexError(f"index {index} out of range (0, {len(self.__instances) - 1})")

        del self.__instances[index]

    def __add__(self, other: 'Cluster') -> 'Cluster':
        """
        Concatenates two clusters together with the operator + and updates the centroid

        Parameters:
            * other: `Cluster`. The cluster to concatenate

        Returns:
            `Cluster`. The concatenated cluster

        Raises:
            TypeError: if the parameter is not an instance of `Cluster`
            ValueError: if the clusters do not have the same length of instances in each `Subsequence`

        Examples:
            >>> sequence1 = Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence2 = Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 1))
            >>> cluster1 = Cluster(np.array([3, 4, 5, 6]), sequence1)
            >>> cluster2 = Cluster(np.array([7, 8, 9, 10]), sequence2)
            >>> new_cluster = cluster1 + cluster2
            >>> print(new_cluster)
            Cluster(
                -Centroid: np.array([5, 6, 7, 8])
                -Instances: [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])]
                -Dates: [datetime.date(2021, 1, 1), datetime.date(2021, 1, 2)]
                -Starting Points: [0, 1]
            )
        """

        if not isinstance(other, Cluster):
            raise TypeError(f"other must be an instance of Cluster. Got {type(other).__name__} instead")

        # Check if the lengths of the subsequences from the instances of each cluster match
        if len(self.__instances[0]) != len(other.get_sequences()[0]):
            raise ValueError(
                f"clusters do not have the same length of instances in each Subsequence. Expected len={len(self.__instances[0])} but got len={len(other.get_sequences()[0])}")

        new_instances = self.__instances + other.get_sequences()
        new_centroid = np.mean(new_instances.get_subsequences(), axis=0)
        return Cluster(centroid=new_centroid, instances=new_instances)

    def __eq__(self, other: Union['Cluster', None]) -> bool:
        """
        Check if the cluster is equal to another cluster with the operator ==

        Parameters:
            * other: `Cluster`. The cluster to check

        Returns:
            `bool`. `True` if the clusters are equal, `False` otherwise

        Raises:
            TypeError: if the parameter is not an instance of `Cluster`

        Examples:
            >>> sequence1 = Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence2 = Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 1))
            >>> cluster1 = Cluster(np.array([3, 4, 5, 6]), sequence1)
            >>> cluster2 = Cluster(np.array([3, 4, 5, 6]), sequence2)
            >>> cluster1 == cluster2
            False

            >>> sequence3 = Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence4 = Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 1))
            >>> cluster3 = Cluster(np.array([3, 4, 5, 6]), sequence3)
            >>> cluster4 = Cluster(np.array([3, 4, 5, 6]), sequence4)
            >>> cluster3 == cluster4
            True
        """

        # Check if the other is a Cluster instance
        if not isinstance(other, Cluster):
            if other is None:
                return False

            raise TypeError(f"other must be an instance of Cluster. Got {type(other).__name__} instead")

        # Check if the centroid and the instances are equal
        if not np.array_equal(self.__centroid, other.centroid):
            return False

        # Check if the number of instances is equal
        if len(self.__instances) != len(other.get_sequences()):
            return False

        # Check if the length of the instances is equal
        if self.__length != other.length_cluster_subsequences:
            return False

        # Check if the instances are equal
        if not np.array_equal(self.__instances.get_subsequences(), other.get_sequences().get_subsequences()):
            return False

        # Check if the dates are equal
        if not np.array_equal(self.__instances.get_dates(), other.get_sequences().get_dates()):
            return False

        # Check if the starting points are equal
        if not np.array_equal(self.__instances.get_starting_points(), other.get_sequences().get_starting_points()):
            return False

        return True

    @staticmethod
    def __check_validity_params(centroid: np.ndarray, instances: 'Sequence') -> None:
        """
        Check if the parameters are valid

        Parameters:
            * centroid: `np.ndarray`. The centroid of the cluster
            * instances: `Sequence`. The sequence of subsequences

        Raises:
            TypeError: if the parameters are not of the correct type
            ValueError: if the length of the centroid is not the same as the length of the subsequences
        """

        # Check if the centroid is an instance of np.ndarray
        if not isinstance(centroid, np.ndarray):
            raise TypeError(f"centroid must be an instance of np.ndarray. Got {type(centroid).__name__}")

        # Check if the instances is an instance of Sequence
        if not isinstance(instances, Sequence):
            raise TypeError(f"instances must be an instance of Sequence. Got {type(instances).__name__}")

        # Check if the length of the centroid is the same as the length of the subsequences
        if len(centroid) != instances.length_subsequences:
            raise ValueError(
                f"The length of the centroid must be equal to the length of the subsequences. Got {len(centroid)} and {instances.length_subsequences} instead")

    @property
    def centroid(self) -> np.ndarray:
        """
        Returns the centroid of the cluster
        :return: np.ndarray. The centroid of the cluster
        """
        return self.__centroid

    @centroid.setter
    def centroid(self, subsequence: np.ndarray | Subsequence) -> None:
        """
        Sets the value of the centroid from the cluster with a subsequence

        Parameters:
            * subsequence: `Union[Subsequence|np.ndarray]`. The subsequence to set as the centroid

        Raises:
            TypeError: if the parameter is not a `Subsequence` or a numpy array
            ValueError: if the length of the subsequence is not the same as the length of the subsequences
        """

        # Check if the length of the subsequence is the same as the length of the subsequences
        if len(subsequence) != self.__length:
            raise ValueError(f"the length of the subsequence must be {self.__length}. Got {len(subsequence)} instead")

        # Set the centroid if it is an instance of Subsequence
        if isinstance(subsequence, Subsequence):
            self.__centroid = subsequence.get_instance()

        # Set the centroid if it is a numpy array
        if isinstance(subsequence, np.ndarray):
            self.__centroid = subsequence

        # Raise an error if the parameter is not a Subsequence or a numpy array
        if not isinstance(subsequence, Subsequence) and not isinstance(subsequence, np.ndarray):
            raise TypeError(
                f"subsequence must be an instance of Subsequence or a numpy array. Got {type(subsequence).__name__} instead")

    @property
    def length_cluster_subsequences(self) -> int:
        """
        Getter that returns the length of the instances in the cluster

        Returns:
            `int`. The length of the instances in the cluster

        Examples:
            >>> sequence = Sequence(Subsequence(np.array([1, 2, 3, 4, 5]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8, 9]), datetime.date(2021, 1, 2), 4))
            >>> cluster = Cluster(np.array([3, 4, 5, 6, 7]), sequence)
            >>> cluster.length_cluster_subsequences
            5
        """

        return self.__length

    def add_instance(self, new_instance: 'Subsequence') -> None:
        """
        Adds a subsequence to the instances of the cluster

        Parameters:
            * new_instance: `Subsequence`. The subsequence to add

        Raises:
            TypeError: if the parameter is not of the correct type
            ValueError: if the subsequence is already an instance of the cluster

        Examples:
            >>> sequence = Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> cluster = Cluster(np.array([3, 4, 5, 6]), sequence)
            >>> cluster.add_instance(Subsequence(np.array([9, 10, 11, 12]), datetime.date(2021, 1, 3), 0))
            >>> print(cluster)
            Cluster(
                -Centroid: np.array([3, 4, 5, 6])
                -Instances: [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8]), np.array([9, 10, 11, 12])]
                -Dates: [datetime.date(2021, 1, 1), datetime.date(2021, 1, 2), datetime.date(2021, 1, 3)]
                -Starting Points: [0, 4, 0]
            )
        """

        # Check if the new_sequence is a Subsequence instance
        if not isinstance(new_instance, Subsequence):
            raise TypeError(
                f"new sequence must be an instance of Subsequence. Got {type(new_instance).__name__} instead")

        # Check if the new sequence is already an instance of the cluster
        if self.__instances._already_exists(new_instance):
            raise ValueError(f"new sequence is already an instance of the cluster")

        # Check if the length of the new instance is the same as the length of the subsequences
        if len(new_instance) != self.__length:
            raise ValueError(
                f"the length of the subsequence must be {self.__length}. Got {len(new_instance)} instead")

        self.__instances.add_sequence(new_instance)

    def get_sequences(self) -> 'Sequence':
        """
        Returns the sequence of the cluster

        Returns:
             `Sequence`. The sequence of the cluster

        Examples:
            >>> sequence = Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> cluster = Cluster(np.array([3, 4, 5, 6]), sequence)
            >>> cluster.get_sequences()
            Sequence(
                length_subsequences=4,
                list_sequences=[
                    Subsequence(
                        instance=np.array([1, 2, 3, 4]),
                        date=datetime.date(2021, 1, 1),
                        starting_point=0
                    ),
                    Subsequence(
                        instance=np.array([5, 6, 7, 8]),
                        date=datetime.date(2021, 1, 2),
                        starting_point=4
                    )
                ]
            )
        """

        return self.__instances

    def update_centroid(self) -> None:
        """
        Updates the centroid of the cluster with the mean of the instances
        """

        self.__centroid = np.mean(self.__instances.get_subsequences(), axis=0)

    def get_starting_points(self, to_array: bool = False) -> Union[list[int], np.ndarray]:
        """
        Returns the starting points of the subsequences

        Parameters:
            * to_array: `bool`. If True, the starting points are returned as a numpy array. Default is False

        Returns:
             `list[int] | np.ndarray`. The starting points of the subsequences
        """

        if to_array:
            return np.array(self.__instances.get_starting_points())

        return self.__instances.get_starting_points()

    def get_dates(self) -> list[datetime.date]:
        """
        Returns the dates of the subsequences

        Returns:
             `list[datetime.date]`. The dates of the subsequences
        """

        return self.__instances.get_dates()

    def cumulative_magnitude(self) -> float | int:
        """
        Returns the magnitude's sum of the subsequences that belongs to the instances within the cluster

        Returns:
             `float`. The magnitude's sum of the subsequences
        """

        return sum([subsequence.magnitude() for subsequence in self.__instances])


class Routines:
    """
    Represents a collection of clusters, each of them representing a routine.

    Parameters:
    _________
        * ``cluster: Optional[Cluster]``, the cluster to add to the collection. Default is None

    Properties:
    ________
        **Getters**:
            * ``hierarchy: int``, the length of the subsequences in the clusters

    Public Methods:
    _________
        * ``add_routine(new_routine: Cluster)``: adds a cluster to the collection
        * ``drop_indexes(to_drop: list[int])``: drops the clusters at the specified indexes
        * ``get_routines() -> list[Cluster]``: returns the routines as a list of clusters
        * ``get_centroids() -> list[np.ndarray]``: returns the centroids of the clusters as a list of numpy arrays
        * ``to_collection() -> list[dict]``: returns the routines as a list of dictionaries
        * ``is_empty() -> bool``: checks if the collection is empty

    Examples:

        >>> subsequence1 = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
        >>> subsequence2 = Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)

        >>> sequence = Sequence(subsequence=subsequence1)
        >>> cluster1 = Cluster(np.array([1, 1, 1, 1]), sequence)

        >>> sequence = Sequence(subsequence=subsequence2)
        >>> cluster2 = Cluster(np.array([5, 5, 5, 5]), sequence)

        >>> routines = Routines(cluster=cluster1)
        >>> routines.add_routine(cluster2)

        >>> routines.get_routines()
        [Cluster(
            centroid=np.array([1, 1, 1, 1]),
            sequences=Sequence(
                length_subsequences=4,
                list_sequences=[
                    Subsequence(
                        instance=np.array([1, 2, 3, 4]),
                        date=datetime.date(2021, 1, 1),
                        starting_point=0
                    )
                ]
            )
        ), Cluster(
            centroid=np.array([5, 5, 5, 5]),
            sequences=Sequence(
                length_subsequences=4,
                list_sequences=[
                    Subsequence(
                        instance=np.array([5, 6, 7, 8]),
                        date=datetime.date(2021, 1, 2),
                        starting_point=4
                    )
                ]
            )
        )]

        >>> routines.get_centroids()
        [np.array([1, 1, 1, 1]), np.array([5, 5, 5, 5])]

        >>> routines.to_collection()
        [{'centroid': np.array([1, 1, 1, 1]), 'sequences': [{'instance': np.array([1, 2, 3, 4]), 'date': datetime.date(2021, 1, 1), 'starting_point': 0}], 'length_subsequences': 4},
         {'centroid': np.array([5, 5, 5, 5]), 'sequences': [{'instance': np.array([5, 6, 7, 8]), 'date': datetime.date(2021, 1, 2), 'starting_point': 4}], 'length_subsequences': 4}]

        >>> routines.is_empty()
        False

        >>> routines.drop_indexes([0])
        >>> routines.get_routines()
        [Cluster(
            centroid=np.array([5, 5, 5, 5]),
            sequences=Sequence(
                length_subsequences=4,
                list_sequences=[
                    Subsequence(
                        instance=np.array([5, 6, 7, 8]),
                        date=datetime.date(2021, 1, 2),
                        starting_point=4
                    )
                ]
            )
        )]
    """

    def __init__(self, cluster: Optional[Cluster] = None) -> None:
        """
        Parameters:
            * cluster: `Optional[Cluster]`, the cluster to add to the routines. Default is None

        Raises:
             TypeError: if the parameter is not an instance of cluster

        Examples:
             >>> routines = Routines()
             >>> print(routines)
             Routines(
                list_routines = [[]]
             )

             >>> sequence = Sequence(subsequence=Subsequence(np.array([1,2,3], datetime.date(2024, 1, 1), 1)))
             >>> routines = Routines(Cluster(centroid=np.array([1,2,3], instances=sequence)))
             >>> print(routines)
             Routines(
                list_routines = [
                    Cluster(
                        - centroid = [1,2,3],
                        - instances = [[1,2,3]]
                        - starting_points = [1]
                        - dates = [datetime.date(2024, 1, 1)]
                    )
                ]
             )
        """

        if cluster is not None:
            if not isinstance(cluster, Cluster):
                raise TypeError(f"cluster has to be an instance of Cluster. Got {type(cluster).__name__} instead")

            self.__routines: list[Cluster] = [cluster]
            self.__hierarchy = cluster.length_cluster_subsequences

        else:
            self.__routines: list[Cluster] = []
            self.__hierarchy = None

    def __repr__(self):
        """
        Returns the string representation of the routines

        Returns:
            `str`. The string representation of the routines

        Examples:
            >>> routines = Routines()
            >>> cluster1 = Cluster(np.array([3, 4, 5, 6]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> cluster2 = Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> routines.add_routine(cluster1)
            >>> routines.add_routine(cluster2)
            >>> print(routines)
            Routines(
                list_routines=[
                    Cluster(
                        - centroid = [3, 4, 5, 6],
                        - instances = [[1, 2, 3, 4]]
                        - starting_points = [0]
                        - dates = [datetime.date(2021, 1, 1)]
                    ),
                    Cluster(
                        - centroid = [7, 8, 9, 10],
                        - instances = [[5, 6, 7, 8]]
                        - starting_points = [4]
                        - dates = [datetime.date(2021, 1, 2)]
                    )
                ]
            )
        """

        out_string = "Routines(\n\tlist_routines=[\n"
        for routine in self.__routines:
            out_string += f" {routine},\n"

        out_string = out_string[:-2] + out_string[-1] + "])"
        return out_string

    def __str__(self):
        """
        Returns the string representation of the routines

        Returns:
            `str`. The string representation of the routines

        Examples:
            >>> routines = Routines()
            >>> cluster1 = Cluster(np.array([3, 4, 5, 6]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> cluster2 = Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> routines.add_routine(cluster1)
            >>> routines.add_routine(cluster2)
            >>> print(routines)
            Routines(
                list_routines=[
                    Cluster(
                        - centroid = [3, 4, 5, 6],
                        - instances = [[1, 2, 3, 4]]
                        - starting_points = [0]
                        - dates = [datetime.date(2021, 1, 1)]
                    ),
                    Cluster(
                        - centroid = [7, 8, 9, 10],
                        - instances = [[5, 6, 7, 8]]
                        - starting_points = [4]
                        - dates = [datetime.date(2021, 1, 2)]
                    )
                ]
            )
        """

        out_string = "Routines(\n\tlist_routines=[\n"
        for routine in self.__routines:
            out_string += f" {routine},\n"

        out_string = out_string[:-2] + out_string[-1] + "])"
        return out_string

    def __len__(self) -> int:
        """
        Returns the number of clusters in the `Routines`

        Returns:
            `int`. The number of clusters in the `Routines`

        Examples:
            >>> routines = Routines()
            >>> cluster1 = Cluster(np.array([3, 4, 5, 6]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> cluster2 = Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> routines.add_routine(cluster1)
            >>> routines.add_routine(cluster2)
            >>> len(routines)
            2
        """

        return len(self.__routines)

    def __getitem__(self, index: int) -> 'Cluster':
        """
        Get the cluster at the specified index in the collection

        Parameters:
            * index: `int`. The index of the cluster

        Returns:
            `Cluster`. The cluster at the specified index in the collection

        Raises:
            TypeError: if the index is not an integer
            IndexError: if the index is out of range of the routines

        Examples:
            >>> routines = Routines()
            >>> cluster1 = Cluster(np.array([3, 4, 5, 6]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> cluster2 = Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> routines.add_routine(cluster1)
            >>> routines.add_routine(cluster2)
            >>> print(routines[0])
            Cluster(
                - centroid = [3, 4, 5, 6],
                - instances = [[1, 2, 3, 4]]
                - starting_points = [0]
                - dates = [datetime.date(2021, 1, 1)]
            )
        """
        # Check if the index is an integer
        if not isinstance(index, int):
            raise TypeError(f"index must be an integer. Got {type(index).__name__} instead")

        # Check if the index is within the range of the list
        if not 0 <= index < len(self.__routines):
            raise IndexError(f"index {index} out of range (0, {len(self.__routines) - 1})")

        return self.__routines[index]

    def __setitem__(self, index: int, value: 'Cluster') -> None:
        """
        Set the value of the cluster at the specified index in the collection

        Parameters:
            * index: `int`. The index of the cluster
            * value: `Cluster`. The new cluster

        Raises:
            TypeError: if the index is not an integer or the value is not an instance of Cluster
            IndexError: if the index is out of range of the routines

        Examples:
            >>> routines = Routines()
            >>> cluster1 = Cluster(np.array([3, 4, 5, 6]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> cluster2 = Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> routines.add_routine(cluster1)
            >>> routines.add_routine(cluster2)
            >>> routines[0] = Cluster(np.array([11, 12, 13, 14]), Sequence(Subsequence(np.array([9, 10, 11, 12]), datetime.date(2021, 1, 3), 4))
            >>> print(routines[0])
            Cluster(
                - centroid = [11, 12, 13, 14],
                - instances = [[9, 10, 11, 12]]
                - starting_points = [4]
                - dates = [datetime.date(2021, 1, 3)]
            )
        """
        # Check if the value is a Cluster instance
        if not isinstance(value, Cluster):
            raise TypeError(f"value has to be an instance of Cluster. Got {type(value).__name__} instead")

        # Check if the index is an integer
        if not isinstance(index, int):
            raise TypeError(f"index has to be an integer. Got {type(index).__name__} instead")

        # Check if the index is within the range of the list
        if not 0 <= index < len(self.__routines):
            raise IndexError(f"index {index} out of range. Got (0, {len(self.__routines) - 1})")

        self.__routines[index] = value

    def __iter__(self) -> iter:
        """
        Returns an iterator for each cluster in the `Routines`

        Returns:
            iter. An iterator for each cluster in the `Routines`

        Examples:
            >>> routines = Routines()
            >>> cluster1 = Cluster(np.array([3, 4, 5, 6]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> cluster2 = Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> routines.add_routine(cluster1)
            >>> routines.add_routine(cluster2)
            >>> for cluster in routines:
            ...     print(cluster)
            Cluster(
                - centroid = [3, 4, 5, 6],
                - instances = [[1, 2, 3, 4]]
                - starting_points = [0]
                - dates = [datetime.date(2021, 1, 1)]
            )
            Cluster(
                - centroid = [7, 8, 9, 10],
                - instances = [[5, 6, 7, 8]]
                - starting_points = [4]
                - dates = [datetime.date(2021, 1, 2)]
            )
        """

        return iter(self.__routines)

    def __contains__(self, item: 'Cluster') -> bool:
        """
        Check if the cluster exists in the collection

        Parameters:
            * item: `Cluster`. The cluster to check

        Returns:
            `bool`. `True` if the cluster exists, `False` otherwise

        Raises:
            TypeError: if the parameter is not an instance of Cluster
        """

        # Check if the item is a Cluster instance
        if not isinstance(item, Cluster):
            raise TypeError(f"item has to be an instance of Cluster. Got {type(item).__name__} instead")

        return item in self.__routines

    def __delitem__(self, index: int) -> None:
        """
        Deletes the cluster at the specified index in the collection

        Parameters:
            * index: `int`. The index of the cluster to delete

        Returns:
            `Cluster`. The cluster at the specified index in the collection

        Raises:
            TypeError: if the index is not an integer
            IndexError: if the index is out of range of the routines
        """

        # Check if the index is an integer
        if not isinstance(index, int):
            raise TypeError(f"index has to be an integer. Got {type(index).__name__} instead")

        # Check if the index is within the range of the list
        if not 0 <= index < len(self.__routines):
            raise IndexError(f"index {index} out of range (0, {len(self.__routines) - 1})")

        del self.__routines[index]

    def __add__(self, other: 'Routines') -> 'Routines':
        """
        Concatenates two routines together with the operator + and returns a new collection

        Parameters:
            * other: `Routines`. The collection to concatenate

        Returns:
            `Routines`. The concatenated `Routines`

        Raises:
            TypeError: if the parameter is not an instance of Routines
            ValueError: if the hierarchy of the routines is not the same

        Examples:
            >>> routines1 = Routines()
            >>> cluster1 = Cluster(np.array([3, 4, 5, 6]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> cluster2 = Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> routines1.add_routine(cluster1)
            >>> routines1.add_routine(cluster2)
            >>> routines2 = Routines()
            >>> cluster3 = Cluster(np.array([11, 12, 13, 14]), Sequence(Subsequence(np.array([9, 10, 11, 12]), datetime.date(2021, 1, 3), 4))
            >>> routines2.add_routine(cluster3)
            >>> new_routines = routines1 + routines2
            >>> print(new_routines)
            Routines(
                list_routines=[
                    Cluster(
                        - centroid = [3, 4, 5, 6],
                        - instances = [[1, 2, 3, 4]]
                        - starting_points = [0]
                        - dates = [datetime.date(2021, 1, 1)]
                    ),
                    Cluster(
                        - centroid = [7, 8, 9, 10],
                        - instances = [[5, 6, 7, 8]]
                        - starting_points = [4]
                        - dates = [datetime.date(2021, 1, 2)]
                    ),
                    Cluster(
                        - centroid = [11, 12, 13, 14],
                        - instances = [[9, 10, 11, 12]]
                        - starting_points = [4]
                        - dates = [datetime.date(2021, 1, 3)]
                    )
                ]
            )
        """
        # Check if the other is a Routines instance
        if not isinstance(other, Routines):
            raise TypeError(f"other has to be an instance of Routines. Got {type(other).__name__} instead")

        # Check if the routines are empty
        if other.is_empty() and not self.is_empty():
            return self

        # Check if the routines are empty
        if not other.is_empty() and self.is_empty():
            return other

        # Check if the hierarchy is the same
        if not other.is_empty() and self.__hierarchy != other[0].length_cluster_subsequences:
            raise ValueError(
                f"the hierarchy of the routines must be the same. Expected {self.__hierarchy}, got {other.__hierarchy} instead")

        # Concatenate the routines if both are not empty
        new_routines = Routines()
        new_routines.__routines = self.__routines + other.__routines
        new_routines.__hierarchy = self.__hierarchy
        return new_routines

    def __eq__(self, other: 'Routines') -> bool:
        """
        Check if the self routine is equal to another routine with the operator ==

        Parameters:
            * other: `Routines`. The routine to check

        Returns:
            `bool`. `True` if the routines are equal, `False` otherwise

        Raises:
            TypeError: if the parameter is not an instance of `Routines`

        Examples:
            >>> routines1 = Routines()
            >>> cluster1 = Cluster(np.array([3, 4, 5, 6]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> cluster2 = Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> routines1.add_routine(cluster1)
            >>> routines1.add_routine(cluster2)
            >>> routines2 = Routines()
            >>> cluster3 = Cluster(np.array([11, 12, 13, 14]), Sequence(Subsequence(np.array([9, 10, 11, 12]), datetime.date(2021, 1, 3), 4))
            >>> routines2.add_routine(cluster3)
            >>> routines1 == routines2
            False

            >>> routines3 = Routines()
            >>> cluster4 = Cluster(np.array([3, 4, 5, 6]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> cluster5 = Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> routines3.add_routine(cluster4)
            >>> routines3.add_routine(cluster5)
            >>> routines1 == routines3
            True
        """

        # Check if the other is a Routines instance
        if not isinstance(other, Routines):
            raise TypeError(f"other has to be an instance of Routines. Got {type(other).__name__} instead")

        # Check if the number of clusters is equal
        if len(self.__routines) != len(other.__routines):
            return False

        if self.__hierarchy != other.__hierarchy:
            return False

        # Check if the clusters are equal
        for idx, routine in enumerate(self.__routines):
            if routine != other.__routines[idx]:
                return False

        return True

    @property
    def hierarchy(self) -> int:
        """
        Returns the hierarchy of the routines

        Returns:
            `int`. The hierarchy of the routines
        """

        return self.__hierarchy

    def add_routine(self, new_routine: 'Cluster') -> None:
        """
        Adds a cluster to the collection

        Parameters:
            new_routine: `Cluster`. The cluster to add

        Raises:
             TypeError: if the parameter is not of the correct type

        Examples:
            >>> routines = Routines()
            >>> cluster1 = Cluster(np.array([3, 4, 5, 6]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> cluster2 = Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> routines.add_routine(cluster1)
            >>> routines.add_routine(cluster2)
            >>> print(routines)
            Routines(
                list_routines=[
                    Cluster(
                        - centroid = [3, 4, 5, 6],
                        - instances = [[1, 2, 3, 4]]
                        - starting_points = [0]
                        - dates = [datetime.date(2021, 1, 1)]
                    ),
                    Cluster(
                        - centroid = [7, 8, 9, 10],
                        - instances = [[5, 6, 7, 8]]
                        - starting_points = [4]
                        - dates = [datetime.date(2021, 1, 2)]
                    )
                ]
            )
        """
        # Check if the new_routine is a Cluster instance
        if not isinstance(new_routine, Cluster):
            raise TypeError(f"new_routine has to be an instance of Cluster. Got {type(new_routine).__name__} instead")

        # Check if the hierarchy is not initialized
        if self.__hierarchy is None:
            self.__hierarchy = new_routine.length_cluster_subsequences

        # Check if the length of the subsequences is the same as the hierarchy
        if new_routine.length_cluster_subsequences != self.__hierarchy:
            raise ValueError(
                f"the length of the subsequences must be {self.__hierarchy}. Got {new_routine.length_cluster_subsequences} instead")

        self.__routines.append(new_routine)

    def drop_indexes(self, to_drop: list[int]) -> 'Routines':
        """
        Drops the clusters with the specified indexes

        Parameters:
            to_drop: `list[int]`. The indexes of the clusters to drop

        Returns:
             Routines. The collection without the dropped clusters

        Examples:
            >>> routines = Routines()
            >>> cluster1 = Cluster(np.array([3, 4, 5, 6]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)))
            >>> cluster2 = Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 2)))
            >>> cluster3 = Cluster(np.array([11, 12, 13, 14]), Sequence(Subsequence(np.array([9, 10, 11, 12]), datetime.date(2021, 1, 3), 4)))
            >>> routines.add_routine(cluster1)
            >>> routines.add_routine(cluster2)
            >>> routines.add_routine(cluster3)
            >>> filtered_routines = routines.drop_indexes([0, 2])
            >>> print(filtered_routines)
            Routines(
                list_routines=[
                    Cluster(
                        - centroid = [11, 12, 13, 14],
                        - instances = [[9, 10, 11, 12]]
                        - starting_points = [4]
                        - dates = [datetime.date(2021, 1, 3)]
                    )
                ]
            )

        Notes:
            This method does not modify the original `Routine`, it returns a new one without the dropped clusters
        """

        new_routines = Routines()
        for idx, cluster in enumerate(self.__routines):
            if idx not in to_drop:
                new_routines.add_routine(cluster)
        return new_routines

    def get_routines(self) -> list[Cluster]:
        """
        Returns the clusters of the `Routines`

        Returns
            `list[Cluster]`. Returns all the clusters of the routines as a list of clusters

        Examples:
            >>> routines = Routines()
            >>> cluster1 = Cluster(np.array([3, 4, 5, 6]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> cluster2 = Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> routines.add_routine(cluster1)
            >>> routines.add_routine(cluster2)
            >>> routines.get_routines()
            [Cluster(centroid=np.array([3, 4, 5, 6]), instances=Sequence(list_sequences=[Subsequence(instance=np.array([1, 2, 3, 4]), date=datetime.date(2021, 1, 1), starting_point=0)]), Cluster(centroid=np.array([7, 8, 9, 10]), instances=Sequence(list_sequences=[Subsequence(instance=np.array([5, 6, 7, 8]), date=datetime.date(2021, 1, 2), starting_point=4)])]
        """

        return self.__routines

    def get_centroids(self) -> list[np.ndarray]:
        """
        Returns the centroids of the clusters

        Returns:
             `list[np.ndarray]`. The centroids of the clusters

        Examples:
            >>> routines = Routines()
            >>> cluster1 = Cluster(np.array([3, 4, 5, 6]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> cluster2 = Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> routines.add_routine(cluster1)
            >>> routines.add_routine(cluster2)
            >>> routines.get_centroids()
            [np.array([3, 4, 5, 6]), np.array([7, 8, 9, 10])]
        """

        return [cluster.centroid for cluster in self.__routines]

    def to_collection(self) -> list[dict]:
        """
        Returns the collection as a list of dictionaries

        Returns:
             `list[dict]`. The routines as a list of dictionaries

        Examples:
            >>> routines = Routines()
            >>> cluster1 = Cluster(np.array([3, 4, 5, 6]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> cluster2 = Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> routines.add_routine(cluster1)
            >>> routines.add_routine(cluster2)
            >>> routines.to_collection()
            [{ 'centroid': np.array([3, 4, 5, 6]),
              'instances': [{'instance': np.array([1, 2, 3, 4]), 'date': datetime.date(2021, 1, 1), 'starting_point': 0}]},
              {'centroid': np.array([7, 8, 9, 10]),
               'instances': [{'instance': np.array([5, 6, 7, 8]), 'date': datetime.date(2021, 1, 2), 'starting_point': 4}]}
            ]
        """

        collection = []
        for routine in self.__routines:
            collection.append({
                'centroid': routine.centroid,
                'instances': routine.get_sequences().to_collection()
            })
        return collection

    def is_empty(self) -> bool:
        """
        Returns `True` if routines is empty, `False` otherwise

        Returns:
            `bool`. `True` if the collection is empty, `False` otherwise

        Examples:
            >>> routines = Routines()
            >>> routines.is_empty()
            True

            >>> cluster = Cluster(np.array([3, 4, 5, 6]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> routines.add_routine(cluster)
            >>> routines.is_empty()
            False
        """

        return len(self.__routines) == 0

    def drop_duplicates(self) -> 'Routines':
        """
        Filter the repeated clusters in the routines and returns

        Returns:
            `Routines`. The routines without the repeated clusters
        """

        new_routines = Routines()

        for routine in self.__routines:
            if routine not in new_routines:
                new_routines.add_routine(routine)
        return new_routines


class HierarchyRoutine:
    """
    Represents a hierarchy of routines,
    where the hierarchy corresponds to the length of the subsequences for each cluster in the routine.
    For each hierarchy, exists one routine with the correspondent length of the subsequences.

    Parameters:
    _________
        * ``routine: Optional[Routines]``, the routine to add to the hierarchy. Default is None

    Public Methods:
    _______________

        * ``add_routine(new_routine: Routines)``: adds a routine to the hierarchy
        * ``to_dictionary() -> dict``: returns the hierarchy as a dictionary

    Properties:
    ___________
        **Getters:**
            * ``keys: list[int]``: returns the list with all the hierarchies registered
            * ``values: list[Routines]``: returns a list with all the routines
            * ``items: Iterator[tuple[int, Routines]]``: returns a iterator as a zip object with the hierarchy and the routine

    Examples:

            >>> subsequence1 = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> subsequence2 = Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)

            >>> sequence = Sequence(subsequence=subsequence1)
            >>> cluster1 = Cluster(np.array([1, 1, 1, 1]), sequence)

            >>> sequence = Sequence(subsequence=subsequence2)
            >>> cluster2 = Cluster(np.array([5, 5, 5, 5]), sequence)

            >>> routines = Routines(cluster=cluster1)
            >>> routines.add_routine(cluster2)

            >>> hierarchy = HierarchyRoutine(routines=routines)
            >>> hierarchy.to_dictionary()
            {4: [{'centroid': array([1, 1, 1, 1]), 'instances': [{'instance': array([1, 2, 3, 4]), 'date': datetime.date(2021, 1, 1), 'starting_point': 0}]},
                 {'centroid': array([5, 5, 5, 5]), 'instances': [{'instance': array([5, 6, 7, 8]), 'date': datetime.date(2021, 1, 2), 'starting_point': 4}]}]}

            >>> hierarchy.keys
            [4]

            >>> hierarchy.values
            [Routines(
                list_routines=[
                  Cluster(
                     -Centroid: [1 1 1 1]
                     -Instances: [array([1, 2, 3, 4])]
                     -Dates: [datetime.date(2021, 1, 1)]
                     -Starting Points: [0]
                 ),
                  Cluster(
                     -Centroid: [5 5 5 5]
                     -Instances: [array([5, 6, 7, 8])]
                     -Dates: [datetime.date(2021, 1, 2)]
                     -Starting Points: [4]
                 )
            ])]

            >>> for key, value in hierarchy.items:
            ...     print(key, value)
            4 Routines(
                list_routines=[
                     Cluster(
                         -Centroid: [1 1 1 1]
                         -Instances: [array([1, 2, 3, 4])]
                         -Dates: [datetime.date(2021, 1, 1)]
                         -Starting Points: [0]
                    ),
                     Cluster(
                         -Centroid: [5 5 5 5]
                         -Instances: [array([5, 6, 7, 8])]
                         -Dates: [datetime.date(2021, 1, 2)]
                         -Starting Points: [4]
                    )
            ])
    """

    def __init__(self, routines: Optional[Routines] = None) -> None:
        """
        Initializes the HierarchyRoutine with the routines

        Parameters:
            * routines: ``Optional[Routines]``, the routines to add to the collection. Default is `None`

        Raises:
            TypeError: if the routines is not an instance of `Routines`
            ValueError: if the routines are empty

        Examples:

            >>> sequence = Sequence(Subsequence(np.array([1, 2, 3]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7]), datetime.date(2021, 1, 2), 4))
            >>> cluster = Cluster(np.array([3, 4, 5]), sequence)
            >>> routines = Routines(cluster=cluster)

            >>> hierarchy_routine = HierarchyRoutine(routines)

            >>> hierarchy_routine = HierarchyRoutine()
            >>> hierarchy_routine.add_routine(routines)

            >>> print(hierarchy_routine)
            HierarchyRoutine(
                [Hierarchy: 3.
                    Routines(
                        list_routines=[
                            Cluster(
                                - centroid = [3, 4, 5],
                                - instances = [[1, 2, 3], [5, 6, 7]]
                                - starting_points = [0, 4]
                                - dates = [datetime.date(2021, 1, 1)]
                            )
                        ]
                    )
                ]
            )
        """

        self.__list_routines: list[Routines] = []
        self.__hierarchy: list[int] = []

        # check if a routine is provided
        if routines is not None:
            # Check if the routines is an instance of Routines
            if not isinstance(routines, Routines):
                raise TypeError(f"routines has to be an instance of Routines. Got {type(routines).__name__} instead")

            # Check if the routines are not empty
            if len(routines) == 0:
                raise ValueError("routines cannot be empty")

            # Add the routine to the hierarchy routine
            self.__hierarchy = [routines.hierarchy]
            self.__list_routines.append(routines)

    def __str__(self):
        out_string = "HierarchyRoutine(\n"
        for idx, routine in enumerate(self.__list_routines):
            out_string += f" [Hierarchy: {self.__hierarchy[idx]}. \n\t{routine} ], \n"

        out_string = out_string[:-2] + out_string[-1] + ")"
        return out_string

    def __repr__(self):
        out_string = "HierarchyRoutine(\n"
        for idx, routine in enumerate(self.__list_routines):
            out_string += f" [Hierarchy: {self.__hierarchy[idx]}. \n\t{routine} ], \n"

        out_string = out_string[:-2] + out_string[-1] + ")"
        return out_string

    def __setitem__(self, hierarchy: int, routine: Routines) -> None:
        """
        Sets the routine at the specified hierarchy

        Parameters:
            * hierarchy: `int`. The hierarchy of the routine
            * routine: `Routines`. The routine to set

        Raises:
            TypeError: if the hierarchy is not an integer or the routine is not an instance of Routines
            ValueError: if the routine is empty or the hierarchy is not the same as the routine hierarchy

        Examples:
            >>> routine = Routines(cluster=Cluster(np.array([3, 4, 5]), Sequence(Subsequence(np.array([1, 2, 3]), datetime.date(2021, 1, 1), 0)))
            >>> routine2 = Routines(cluster=Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)))

            >>> hierarchy_routine = HierarchyRoutine(routine)
            >>> hierarchy_routine[4] = routine2
        """
        # Check if the hierarchy is an integer
        if not isinstance(hierarchy, int):
            raise TypeError(f"hierarchy has to be an integer. Got {type(hierarchy).__name__} instead")

        # Check if the routine is an instance of Routines
        if not isinstance(routine, Routines):
            raise TypeError(f"routine has to be an instance of Routines. Got {type(routine).__name__} instead")

        # Check if the routine is empty
        if routine.is_empty():
            raise ValueError("routine cannot be empty")

        # Check if the hierarchy is the same as the routine hierarchy
        if hierarchy != routine.hierarchy:
            raise ValueError(
                f"the hierarchy of the routines must be the same. Expected {hierarchy}. Got {routine.hierarchy} instead")

        # If the hierarchy exists, we update the value
        if hierarchy in self.__hierarchy:
            idx = self.__hierarchy.index(hierarchy)
            self.__list_routines[idx] = routine

        # If the hierarchy doesn't exist, we create a new tuple key, value
        else:
            self.__hierarchy.append(hierarchy)
            self.__list_routines.append(routine)

    def __getitem__(self, hierarchy: int) -> Routines:
        """
        Get the routine at the specified hierarchy

        Parameters:
            * hierarchy: `int`. The hierarchy of the routine

        Returns:
            `Routines`. The routine at the specified hierarchy

        Raises:
            TypeError: if the hierarchy is not an integer
            KeyError: if the hierarchy is not found in the routines

        Examples:
            >>> routine = Routines(cluster=Cluster(np.array([3, 4, 5]), Sequence(Subsequence(np.array([1, 2, 3]), datetime.date(2021, 1, 1), 0)))
            >>> routine2 = Routines(cluster=Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)))

            >>> hierarchy_routine = HierarchyRoutine(routine)
            >>> hierarchy_routine.add_routine(routine2)
            >>> print(hierarchy_routine[3])
            Cluster(
                - centroid = [3, 4, 5],
                - instances = [[1, 2, 3]]
                - starting_points = [0]
                - dates = [datetime.date(2021, 1, 1)]
            )
        """

        # Check if the hierarchy is an integer
        if not isinstance(hierarchy, int):
            raise TypeError(f"hierarchy has to be an integer. Got {type(hierarchy).__name__} instead")

        # Check if the hierarchy exists
        if hierarchy not in self.__hierarchy:
            raise KeyError(f"hierarchy {hierarchy} not found in {self.__hierarchy}")

        # Get the index of the hierarchy
        idx = self.__hierarchy.index(hierarchy)
        return self.__list_routines[idx]

    def __len__(self) -> int:
        """
        Returns the number of routines in the hierarchical routines

        Returns:
            `int`. The number of routines in the hierarchical routines

        Examples:
            >>> routine = Routines(cluster=Cluster(np.array([3, 4, 5]), Sequence(Subsequence(np.array([1, 2, 3]), datetime.date(2021, 1, 1), 0)))
            >>> routine2 = Routines(cluster=Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)))

            >>> hierarchy_routine = HierarchyRoutine(routine)
            >>> hierarchy_routine.add_routine(routine2)
            >>> len(hierarchy_routine)
            2
        """

        return len(self.__list_routines)

    def __contains__(self, routine: Routines) -> bool:
        """
        Check if the routine exists in the hierarchical routines

        Parameters:
            * routine: `Routines`. The routine to check

        Returns:
            `bool`. `True` if the routine exists, `False` otherwise

        Examples:
            >>> routine = Routines(cluster=Cluster(np.array([3, 4, 5]), Sequence(Subsequence(np.array([1, 2, 3]), datetime.date(2021, 1, 1), 0)))
            >>> routine2 = Routines(cluster=Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)))
            >>> hierarchy_routine = HierarchyRoutine(routine)
            >>> hierarchy_routine.add_routine(routine2)
            >>> routine in hierarchy_routine
            True

            >>> routine3 = Routines(cluster=Cluster(np.array([11, 12, 13]), Sequence(Subsequence(np.array([9, 10, 11]), datetime.date(2021, 1, 3), 0)))
            >>> routine3 in hierarchy_routine
            False
        """

        # Check if the routine is an instance of Routines
        if not isinstance(routine, Routines):
            raise TypeError(f"routine has to be an instance of Routines. Got {type(routine).__name__} instead")

        return routine in self.__list_routines

    def add_routine(self, routine: Routines) -> None:
        """
        Adds a routine to the HierarchyRoutine.
        If the key (hierarchy) already exists, it updates the value (Routine).
        If not, it creates a new key, value

        Parameters:
            * routine: `Routines`. The routine to add

        Raises:
            TypeError: if the routine is not an instance of Routines
            ValueError: if the routine is empty

        Examples:
            >>> routine = Routines(cluster=Cluster(np.array([3, 4, 5]), Sequence(Subsequence(np.array([1, 2, 3]), datetime.date(2021, 1, 1), 0)))
            >>> routine2 = Routines(cluster=Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)))

            >>> hierarchy_routine = HierarchyRoutine(routine)
            >>> hierarchy_routine.add_routine(routine2)
            HierachyRoutine(
                [Hierarchy: 3.
                    Routines(
                        list_routines=[
                            Cluster(
                                - centroid = [3, 4, 5],
                                - instances = [[1, 2, 3]]
                                - starting_points = [0]
                                - dates = [datetime.date(2021, 1, 1)]
                            )
                        ]
                    )
                ],
                [Hierarchy: 4.
                    Routines(
                        list_routines=[
                            Cluster(
                                - centroid = [7, 8, 9, 10],
                                - instances = [[5, 6, 7, 8]]
                                - starting_points = [4]
                                - dates = [datetime.date(2021, 1, 2)]
                            )
                        ]
                    )
                ]
            )
        """

        # Check if the routine is an instance of Routines
        if not isinstance(routine, Routines):
            raise TypeError(f"routine has to be an instance of Routines. Got {type(routine).__name__}")

        # Check if the routine is empty
        if routine.is_empty():
            raise ValueError("routine cannot be empty")

        # Get the hierarchy of the routine
        length_clusters = routine.hierarchy

        # If doesn't exist, we create a new tuple key, value
        if length_clusters not in self.__hierarchy:
            self.__list_routines.append(routine)
            self.__hierarchy.append(length_clusters)

        # If it exists, we update the value
        else:
            idx = self.__hierarchy.index(length_clusters)
            self.__list_routines[idx] = routine

    @property
    def keys(self) -> list[int]:
        """
        Returns the hierarchy of the routines

        Returns:
            `list[int]`. The hierarchy of the routines

        Examples:
            >>> routine = Routines(cluster=Cluster(np.array([3, 4, 5]), Sequence(Subsequence(np.array([1, 2, 3]), datetime.date(2021, 1, 1), 0)))
            >>> routine2 = Routines(cluster=Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)))

            >>> hierarchy_routine = HierarchyRoutine(routine)
            >>> hierarchy_routine[4] = routine2
            >>> hierarchy_routine.keys
            [3, 4]
        """

        return self.__hierarchy

    @property
    def values(self) -> list[Routines]:
        """
        Returns the routines

        Returns:
            `list[Routines]`. The routines

        Examples:
            >>> routine = Routines(cluster=Cluster(np.array([3, 4, 5]), Sequence(Subsequence(np.array([1, 2, 3]), datetime.date(2021, 1, 1), 0)))
            >>> routine2 = Routines(cluster=Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)))

            >>> hierarchy_routine = HierarchyRoutine(routine)
            >>> hierarchy_routine[4] = routine2
            >>> hierarchy_routine.values
            [Routines(
                list_routines=[
                    Cluster(
                        - centroid = [3, 4, 5],
                        - instances = [[1, 2, 3]]
                        - starting_points = [0]
                        - dates = [datetime.date(2021, 1, 1)]
                    )
                ]
            ), Routines(
                list_routines=[
                    Cluster(
                        - centroid = [7, 8, 9, 10],
                        - instances = [[5, 6, 7, 8]]
                        - starting_points = [4]
                        - dates = [datetime.date(2021, 1, 2)]
                    )
                ]
            )]
        """

        return self.__list_routines

    @property
    def items(self) -> Iterator[tuple[int, Routines]]:
        """
        Getter that returns the hierarchy and the routines as a zip object iterator

        Returns:
            `Iterator[tuple[int, Routines]]`. The hierarchy and the routines as a zip object iterator

        Examples:
            >>> routine = Routines(cluster=Cluster(np.array([3, 4, 5]), Sequence(Subsequence(np.array([1, 2, 3]), datetime.date(2021, 1, 1), 0)))
            >>> routine2 = Routines(cluster=Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)))

            >>> hierarchy_routine = HierarchyRoutine(routine)
            >>> hierarchy_routine[4] = routine2
            >>> for hierarchy, routine in hierarchy_routine.items:
            ...     print(hierarchy, routine)
            3 Routines(
                list_routines=[
                    Cluster(
                        - centroid = [3, 4, 5],
                        - instances = [[1, 2, 3]]
                        - starting_points = [0]
                        - dates = [datetime.date(2021, 1, 1)]
                    )
                ]
            )
            4 Routines(
                list_routines=[
                    Cluster(
                        - centroid = [7, 8, 9, 10],
                        - instances = [[5, 6, 7, 8]]
                        - starting_points = [4]
                        - dates = [datetime.date(2021, 1, 2)]
                    )
                ]
            )

        """
        return zip(self.__hierarchy, self.__list_routines)

    # def filter_repeated_routines_on_hierarchy(self, hierarchy: int) -> 'HierarchyRoutine':
    #     """
    #     Filters the repeated routines in the correspondent hierarchy
    #
    #     Parameters:
    #         * hierarchy: `int`. The hierarchy to filter the repeated routines
    #
    #     Returns:
    #         `HierarchyRoutine`. The hierarchy routine without the repeated routines
    #
    #     Raises:
    #         TypeError: if the hierarchy is not an integer
    #         KeyError: if the hierarchy is not found in the routines
    #
    #     Examples:
    #         >>> routine = Routines(cluster=Cluster(np.array([3, 4, 5]), Sequence(Subsequence(np.array([1, 2, 3]), datetime.date(2021, 1, 1), 0)))
    #         >>> routine2 = Routines(cluster=Cluster(np.array([3, 4, 5]), Sequence(Subsequence(np.array([1, 2, 3]), datetime.date(2021, 1, 1), 0)))
    #         >>> routine3 = Routines(cluster=Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)))
    #
    #         >>> hierarchy_routine = HierarchyRoutine(routine)
    #         >>> hierarchy_routine[4] = routine2
    #         >>> hierarchy_routine[5] = routine3
    #         >>> hierarchy_routine.filter_repeated_routines_on_hierarchy(4)
    #         HierarchyRoutine(
    #             [Hierarchy: 4.
    #                 Routines(
    #                     list_routines=[
    #                         Cluster(
    #                             - centroid = [3, 4, 5],
    #                             - instances = [[1, 2, 3]]
    #                             - starting_points = [0]
    #                             - dates = [datetime.date(2021, 1, 1)]
    #                         )
    #                     ]
    #                 )
    #             ],
    #             [Hierarchy: 5.
    #                 Routines(
    #                     list_routines=[
    #                         Cluster(
    #                             - centroid = [7, 8, 9, 10],
    #                             - instances = [[5, 6, 7, 8]]
    #                             - starting_points = [4]
    #                             - dates = [datetime.date(2021, 1, 2)]
    #                         )
    #                     ]
    #                 )
    #             ]
    #         )
    #     """
    #
    #     # Check if the hierarchy is an integer
    #     if not isinstance(hierarchy, int):
    #         raise TypeError(f"hierarchy has to be an integer. Got {type(hierarchy).__name__} instead")
    #
    #     # Check if the hierarchy exists
    #     if hierarchy not in self.__hierarchy:
    #         raise KeyError(f"hierarchy {hierarchy} not found in {self.__hierarchy}")
    #
    #     # Get the index of the hierarchy
    #     idx = self.__hierarchy.index(hierarchy)
    #     routine = self.__list_routines[idx]
    #
    #     # Get the subsequences from each cluster in the routine
    #     sequences: list[Sequence] = []
    #     for cluster in routine.get_routines():
    #         sequences.append(cluster.get_sequences())
    #
    #     # Get the unique subsequences
    #     unique_sequences = []
    #     for sequence in sequences:
    #         if sequence not in unique_sequences:
    #             unique_sequences.append(sequence)
    #
    #     # Create a new routine with the unique subsequences
    #     new_routine = Routines()
    #     for sequence in unique_sequences:
    #         new_centroid = sequence.get_subsequences(to_array=True).mean(axis=0)
    #         new_routine.add_routine(Cluster(centroid=new_centroid, instances=sequence))
    #
    #     # Create a new hierarchy routine with the new routine
    #     new_hierarchy_routine = HierarchyRoutine()
    #     new_hierarchy_routine[hierarchy] = new_routine
    #
    #     # Add the other routines to the new hierarchy routine
    #     for key, value in self.items:
    #         if key != hierarchy:
    #             new_hierarchy_routine[key] = value
    #
    #     return new_hierarchy_routine

    def to_dictionary(self) -> dict:
        """
        Returns the routines as a dictionary

        Returns:
            `dict`. The routines as a dictionary

        Examples:
            >>> routine = Routines(cluster=Cluster(np.array([3, 4, 5]), Sequence(Subsequence(np.array([1, 2, 3]), datetime.date(2021, 1, 1), 0)))
            >>> routine2 = Routines(cluster=Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)))

            >>> hierarchy_routine = HierarchyRoutine(routine)
            >>> hierarchy_routine[4] = routine2
            >>> hierarchy_routine.to_dictionary()
            {3: Routines(
                list_routines=[
                    Cluster(
                        - centroid = [3, 4, 5],
                        - instances = [[1, 2, 3]]
                        - starting_points = [0]
                        - dates = [datetime.date(2021, 1, 1)]
                    )
                ]
            ), 4: Routines(
                list_routines=[
                    Cluster(
                        - centroid = [7, 8, 9, 10],
                        - instances = [[5, 6, 7, 8]]
                        - starting_points = [4]
                        - dates = [datetime.date(2021, 1, 2)]
                    )
                ]
            )}
        """

        out_dict = {}
        for idx, hierarchy in enumerate(self.__hierarchy):
            out_dict[hierarchy] = self.__list_routines[idx].to_collection()
        return out_dict


class ClusterNode:
    def __init__(self, cluster: Cluster, depth: int = 0, is_root: bool = False):
        self.__node = cluster
        self.__left: Optional[ClusterNode] = None
        self.__right: Optional[ClusterNode] = None
        self.__depth = depth
        self.__is_root = is_root

    def __str__(self):
        indent = '   ' * self.__depth
        description = f"{indent} ClusterNode(Depth= {self.__depth}, instances = {self.__node.get_sequences().get_subsequences()})\n"
        if self.__left is not None:
            description += "(L)" + self.__left.__str__()
        if self.__right is not None:
            description += "(R)" + self.__right.__str__()

        return description

    def __repr__(self):
        return f"<ClusterNode depth={self.__depth}, instances={self.__node.get_sequences().get_subsequences()}>"

    @property
    def left(self):
        return self.__left

    @left.setter
    def left(self, left: Cluster):
        if not isinstance(left, Cluster):
            raise TypeError(f"left has to be instance of Cluster. Got {type(left).__name__} instead")

        self.__left = ClusterNode(left, self.__depth + 1)

    @property
    def right(self):
        return self.__right

    @right.setter
    def right(self, right):
        if not isinstance(right, Cluster):
            raise TypeError(f"right has to be instance of Cluster. Got {type(right).__name__} instead")

        self.__right = ClusterNode(right, self.__depth + 1)

    @property
    def node(self):
        return self.__node

    @node.setter
    def node(self, cluster):
        if not isinstance(cluster, Cluster):
            raise TypeError(f"cluster has to be instance of Cluster. Got {type(cluster).__name__} instead")

        self.__node = cluster

    @property
    def children(self):
        return [self.__left, self.__right]

    def is_leaf(self) -> bool:
        return self.__left is None and self.__right is None

    def profundity(self) -> int:
        # Check if the node is a leaf
        if self.is_leaf():
            return 1

        return 1 + max(self.__left.profundity(), self.__right.profundity())


class ClusterTree:
    def __init__(self):
        self.__graph: nx.DiGraph = nx.DiGraph()
        self.__nodes: list[Cluster] = []
        self.__list_of_index: list[int] = []

    @staticmethod
    def __convert_edges_to_list(edges: nx.classes.reportviews.OutEdgeDataView) -> list[tuple[int, int, bool]]:
        if not isinstance(edges, nx.classes.reportviews.OutEdgeDataView):
            raise TypeError(f"edges has to be an instance of OutEdgeDataView. Got {type(edges).__name__} instead")

        return list(edges)

    def __check_and_convert_node(self, node: Union[Cluster, int]) -> int:
        if not isinstance(node, (Cluster, int)):
            raise TypeError(
                f"node has to be either an integer or a Cluster instance. Got {type(node).__name__} instead")

        if isinstance(node, Cluster):
            if node not in self.__nodes:
                raise IndexError(f"Cluster {node} not found in the list of clusters {self.__nodes}")

            idx = self.__nodes.index(node)
            return self.__list_of_index[idx]

        return node

    def __assign_edge_color(self) -> list[str]:
        colors = ['red'] * len(self.__graph.edges)
        left = self.__graph.edges.data("left")
        left_formatted = self.__convert_edges_to_list(left)

        for idx, (parent, child, is_left) in enumerate(left_formatted):
            if is_left:
                colors[idx] = "blue"

        return colors

    @property
    def indexes(self):
        return self.__list_of_index

    @property
    def nodes(self):
        return self.__nodes

    @property
    def graph(self):
        return self.__graph

    @property
    def edges(self) -> nx.classes.reportviews.OutEdgeDataView:
        return self.__graph.edges.data()

    def to_dictionary(self) -> dict:
        return dict(zip(self.__list_of_index, self.__nodes))

    def __check_clusters_from_edges(self, parent: Cluster, child: Cluster):
        parent_hierarchy = parent.length_cluster_subsequences
        child_hierarchy = child.length_cluster_subsequences

        if parent not in self.__nodes:
            raise IndexError(f"Cluster {parent} not found in the list of clusters")

        if child not in self.__nodes:
            raise IndexError(f"Cluster {child} not found in the list of clusters")

        if parent == child:
            raise ValueError("The parent and child cannot be the same cluster")

        if child_hierarchy != parent_hierarchy + 1:
            raise ValueError(
                f"The child hierarchy must be the parent hierarchy + 1. Expected {parent_hierarchy + 1}. Got {child_hierarchy} instead")

    def __check_and_return_edge(self, parent: Union[Cluster, int], child: Union[Cluster, int]):
        if not isinstance(parent, (int, Cluster)) or not isinstance(child, (int, Cluster)):
            raise TypeError(
                f"The parent and child must be either an integer or a Cluster instance. Got {type(parent).__name__} and {type(child).__name__} instead")

        if type(parent) != type(child):
            raise TypeError(
                f"The parent and child must be the same type. Got {type(parent).__name__} and {type(child).__name__} instead")

        if isinstance(parent, Cluster):
            self.__check_clusters_from_edges(parent, child)
            return self.get_index_from_cluster(parent), self.get_index_from_cluster(child)

        else:
            if parent not in self.__list_of_index:
                raise IndexError(f"Index {parent} not found in the list of indexes {self.__list_of_index}")

            if child not in self.__list_of_index:
                raise IndexError(f"Index {child} not found in the list of indexes {self.__list_of_index}")

            parent_clust = self.get_cluster_from_index(parent)
            child_clust = self.get_cluster_from_index(child)

            self.__check_clusters_from_edges(parent_clust, child_clust)
            return parent_clust, child_clust

    def get_cluster_from_index(self, idx: int) -> Cluster:

        if not isinstance(idx, int):
            raise TypeError(f"idx has to be an integer. Got {type(idx).__name__} instead")

        if idx not in self.__list_of_index:
            raise IndexError(f"Index {idx} not found in the list of indexes {self.__list_of_index}")

        return self.__graph.nodes.data()[idx]['cluster']

    def get_index_from_cluster(self, node: Cluster) -> int:
        node_index = self.__check_and_convert_node(node)
        return node_index

    def children(self, node: Union[Cluster, int]) -> list[int]:
        node = self.__check_and_convert_node(node)
        edges = list(self.edges)
        children = [child for parent, child, _ in edges if parent == node]

        return children

    def parents(self, node: Union[Cluster, int]) -> list[int]:
        node = self.__check_and_convert_node(node)
        edges = list(self.edges)
        parents = [parent for parent, child, _ in edges if child == node]
        return parents

    def has_children(self, node: Union[Cluster, int]) -> bool:
        return len(self.children(node)) > 0

    def has_parents(self, node: Union[Cluster, int]) -> bool:
        return len(self.parents(node)) > 0

    def assign_node(self, cluster: Cluster):

        if not isinstance(cluster, Cluster):
            raise TypeError(f"cluster has to be an instance of Cluster. Got {type(cluster).__name__} instead")

        if cluster in self.__nodes:
            raise ValueError(f"Cluster {cluster} already exists in the list of clusters {self.__nodes}")

        idx_to_add = len(self.__nodes) + 1

        self.__graph.add_node(idx_to_add, cluster=cluster)
        self.__nodes.append(cluster)
        self.__list_of_index.append(idx_to_add)

    def grow_from_left(self, vertex_parent: Union[Cluster, int], vertex_child: Union[Cluster, int]) -> None:

        parent, child = self.__check_and_return_edge(vertex_parent, vertex_child)
        self.__graph.add_edge(parent, child, left=True, right=False)

    def grow_from_right(self, vertex_parent: Cluster, vertex_child: Cluster):
        parent, child = self.__check_and_return_edge(vertex_parent, vertex_child)
        self.__graph.add_edge(parent, child, left=False, right=True)

    def drop_node(self, node: Union[Cluster, int]):

        def __get_nodes_to_drop_recursive(node: int, nodes_to_drop: list[int]) -> list[int]:

            node: int = self.__check_and_convert_node(node)
            edges = list(self.__graph.edges.data())

            for parent, child, _ in edges:
                if parent == node:
                    if len(self.parents(child)) == 1:
                        nodes_to_drop += __get_nodes_to_drop_recursive(child, [child])

            return nodes_to_drop

        nodes_to_drop = __get_nodes_to_drop_recursive(node, [node])

        for node in nodes_to_drop:
            self.__graph.remove_node(node)
            idx_to_remove = self.__list_of_index.index(node)
            self.__nodes.pop(idx_to_remove)
            self.__list_of_index.pop(idx_to_remove)

    def plot_tree(self, node_size: int = 1000, node_color: str = "lightgray",
                  with_labels: bool = True, figsize: tuple[int, int] = (7, 7),
                  title: Optional[str] = None, title_fontsize: int = 15,
                  save_dir: Optional[str] = None):

        colors = self.__assign_edge_color()
        plt.figure(figsize=figsize)

        if title is not None:
            plt.title(title, fontsize=title_fontsize)

        nx.draw(self.__graph, pos=nx.nx_agraph.graphviz_layout(self.__graph, prog='dot'), with_labels=with_labels,
                node_size=node_size,
                edge_color=colors)

        if save_dir is not None:
            plt.savefig(save_dir)

        plt.show()


#
if __name__ == "__main__":
    cluster31 = Cluster(np.array([3, 4, 5]), Sequence(Subsequence(np.array([1, 2, 3]), datetime.date(2021, 1, 1), 0)))
    cluster32 = Cluster(np.array([5, 6, 7]), Sequence(Subsequence(np.array([5, 6, 7]), datetime.date(2021, 1, 2), 4)))
    cluster33 = Cluster(np.array([10, 11, 12]),
                        Sequence(Subsequence(np.array([10, 11, 12]), datetime.date(2021, 1, 3), 5)))
    cluster41 = Cluster(np.array([15, 16, 17, 18]),
                        Sequence(Subsequence(np.array([15, 16, 17, 18]), datetime.date(2021, 1, 4), 6)))
    cluster42 = Cluster(np.array([20, 21, 22, 23]),
                        Sequence(Subsequence(np.array([20, 21, 22, 23]), datetime.date(2021, 1, 5), 7)))
    cluster51 = Cluster(np.array([25, 26, 27, 28, 29]),
                        Sequence(Subsequence(np.array([25, 26, 27, 28, 29]), datetime.date(2021, 1, 6), 8)))

    tree = ClusterTree()
    tree.assign_node(cluster31)
    tree.assign_node(cluster32)
    tree.assign_node(cluster33)
    tree.assign_node(cluster41)
    tree.assign_node(cluster42)
    tree.assign_node(cluster51)

    tree.grow_from_right(cluster31, cluster41)
    tree.grow_from_right(cluster32, cluster42)
    tree.grow_from_left(cluster33, cluster42)
    tree.grow_from_right(cluster41, cluster51)
    tree.grow_from_left(cluster42, cluster51)

    # print(tree.nodes)
    # print(len(tree.nodes))

    # tree.drop_node(1)
    tree.plot_tree(figsize=(5, 5), title="Cluster Tree")

    # print(tree.graph.nodes.data()[1])
    # print(tree.graph.nodes.data()[2])
    # print(tree.graph.nodes.data()[3])
    # print(type(tree.graph.nodes.data()))

#     cluster_node = ClusterNode(cluster1, is_root=True)
#     cluster_node.left = cluster2
#     cluster_node.right = cluster3
#     cluster_node.left.left = cluster4
#     cluster_node.left.right = cluster5
#     cluster_node.right = cluster6
#     cluster_node.right.right = cluster7
#     # print(cluster_node)
#     print(cluster_node.right)
