import unittest
import sys
import datetime
import numpy as np
import pandas as pd

sys.path.append('..')

from src.structures import Subsequence, Sequence, Cluster, Routines, HierarchyRoutine
from src.DRFL import DRFL, DRGS


class TestSubsequence(unittest.TestCase):
    """
    Test the Subsequence class

    The Subsequence class is a class that represents a subsequence of a time series. It has the following methods:

        * __init__: initializes the Subsequence object
        * distance: returns the distance between two subsequences
        * magnitude: returns the magnitude of the subsequence
        * __len__: returns the length of the subsequence
        * __getitem__: returns the element at the index i
        * __eq__: returns True if the two subsequences are equal, False otherwise
    """

    def setUp(self):
        """
        Set up the Subsequence object and the subsequences for the tests

        The date is 2021-1-1

        The subsequences have the following parameters:

                * Subsequence 1: [1, 2, 3, 4], date: 2021-1-1, starting_point: 0
                * Subsequence 2: [5, 6, 7, 8], date: 2021-1-2, starting_point: 4
                * Subsequence 3: [9, 10, 11, 12], date: 2021-1-3, starting_point: 8
                * Subsequence 4: [13, 14, 15, 16], date: 2021-1-4, starting_point: 12
        """

        self.date = datetime.date(2021, 1, 1)
        self.subsequence1 = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
        self.subsequence2 = Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)
        self.subsequence3 = Subsequence(np.array([9, 10, 11, 12]), datetime.date(2021, 1, 3), 8)
        self.subsequence4 = Subsequence(np.array([13, 14, 15, 16]), datetime.date(2021, 1, 4), 12)

    def test_init(self):
        """
        Test the __init__ method of the Subsequence class

        The method should initialize the Subsequence object with the following parameters:

                * instance: np.array([1, 2, 3, 4])
                * date: 2021-1-1
                * starting_point: 0
        """

        instance = np.array([1, 2, 3, 4])
        starting_point = 0
        self.assertEqual(np.array_equal(self.subsequence1.get_instance(), instance), True)
        self.assertEqual(self.subsequence1.get_date(), self.date)
        self.assertEqual(self.subsequence1.get_starting_point(), starting_point)

    def test_Distance(self):
        """
        Test the distance method of the Subsequence class

        The method should return the distance between two subsequences

        The subsequences are:
                * Subsequence 1: instances=[1, 2, 3, 4], date=2021-1-1, starting_point=0
                * Subsequence 2: instances=[5, 6, 7, 8], date=2021-1-2, starting_point=4

        The expected output is:
                * 4

        Raises:
            * ValueError: if the subsequences have different lengths
            * TypeError: if the other subsequence is not a subsequence or a np.array
        """
        different_length = np.array([1, 2, 3, 4, 5])

        # CASE 1: 4
        self.assertEqual(self.subsequence1.distance(self.subsequence2), 4)

        # CASE 2: ValueError
        with self.assertRaises(ValueError):
            self.subsequence1.distance(different_length)

        # CASE 3: TypeError
        with self.assertRaises(TypeError):
            self.subsequence1.distance("not a subsequence or np.array")

    def test_Magnitude(self):
        """
        Test the magnitude method of the Subsequence class

        The method should return the magnitude of the subsequence

        The subsequences are:
                * Subsequence 1: instances=[1, 2, 3, 4], date=2021-1-1, starting_point=0
                * Subsequence 2: instances=[5, 6, 7, 8], date=2021-1-2, starting_point=4
                * Subsequence 3: instances=[9, 10, 11, 12], date=2021-1-3, starting_point=8
                * Subsequence 4: instances=[13, 14, 15, 16], date=2021-1-4, starting_point=12

        The expected output is:
                * 4 for Subsequence 1
                * 8 for Subsequence 2
                * 12 for Subsequence 3
                * 16 for Subsequence 4
        """

        self.assertEqual(self.subsequence1.magnitude(), 4)
        self.assertEqual(self.subsequence2.magnitude(), 8)
        self.assertEqual(self.subsequence3.magnitude(), 12)
        self.assertEqual(self.subsequence4.magnitude(), 16)

    def test__len__(self):
        """
        Test the __len__ method of the Subsequence class

        The method should return the length of the subsequence

        The subsequence tested is:
                * Subsequence 1: instances=[1, 2, 3, 4], date=2021-1-1, starting_point=0

        The expected output is:
                * 4
        """
        self.assertEqual(len(self.subsequence1), 4)

    def test__getitem__(self):
        """
        Test the __getitem__ method of the Subsequence class

        The method should return the element at the index i

        The subsequence tested is:
                * Subsequence 1: instances=[1, 2, 3, 4], date=2021-1-1, starting_point=0

        The expected output is:
                * 1 for index 0
                * 2 for index 1
                * 3 for index 2
                * 4 for index 3
        """

        self.assertEqual(self.subsequence1[0], 1)
        self.assertEqual(self.subsequence1[1], 2)
        self.assertEqual(self.subsequence1[2], 3)
        self.assertEqual(self.subsequence1[3], 4)

    def test__eq__(self):
        """
        Test the __eq__ method of the Subsequence class

        The method should return True if the two subsequences are equal, False otherwise

        The other subsequences are:
                * first: instances=[1, 2, 3, 4], date=2021-1-1, starting_point=0
                * second: instances=[1, 2, 3, 4], date=2021-1-1, starting_point=1
                * third: instances=[1, 2, 5, 4], date=2021-1-1, starting_point=0
                * fourth: instances=[1, 2, 3, 4], date=2024-1-1, starting_point=0

        The expected subsequence is:
                * Subsequence1: instances=[1, 2, 3, 4], date=2021-1-1, starting_point=0

        The expected output is:
                * True for the first subsequence
                * False for the second subsequence
                * False for the third subsequence
                * False for the fourth subsequence
        """
        # CASE 1: True
        first = Subsequence(np.array([1, 2, 3, 4]), self.date, 0)
        self.assertEqual(self.subsequence1 == first, True)

        # CASE 2: False
        second = Subsequence(np.array([1, 2, 3, 4]), self.date, 1)
        self.assertEqual(self.subsequence1 == second, False)

        # CASE 3: False
        third = Subsequence(np.array([1, 2, 5, 4]), self.date, 0)
        self.assertEqual(self.subsequence1 == third, False)

        # CASE 4: False
        fourth = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2024, 1, 1), 0)
        self.assertEqual(self.subsequence1 == fourth, False)


class TestSequence(unittest.TestCase):
    """
    Test the Sequence class

    The Sequence class is a class that represents a sequence of subsequences. It has the following methods:

            * __init__: initializes the Sequence object
            * add_sequence: adds a subsequence to the sequence
            * get_by_starting_point: returns the subsequence with the starting point t
            * get_starting_points: returns the starting points of the subsequences
            * get_dates: returns the dates of the subsequences
            * get_subsequences: returns the subsequences
    """

    def setUp(self):
        """
        Set up the Sequence object and the subsequences for the tests

        The subsequences have the following parameters:
                    * Subsequence 1: [1, 2, 3, 4], date: 2021-1-1, starting_point: 0
                    * Subsequence 2: [5, 6, 7, 8], date: 2021-1-2, starting_point: 4
        """

        self.sequence = Sequence()
        self.subsequence1 = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
        self.subsequence2 = Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)
        self.other_length = Subsequence(np.array([1, 2, 3, 4, 5]), datetime.date(2021, 1, 3), 8)

    def test_add_sequence(self):
        """
        Test the add_sequence method of the Sequence class

        The method should add a subsequence to the sequence

        The subsequences are:
                * Subsequence 1: instances=[1, 2, 3, 4], date=2021-1-1, starting_point=0
                * Subsequence 2: instances=[5, 6, 7, 8], date=2021-1-2, starting_point=4

        The expected output is:
                * 1 for the length of the sequence after adding the first subsequence
                * 2 for the length of the sequence after adding the second subsequence
        """

        self.sequence.add_sequence(self.subsequence1)
        self.assertEqual(len(self.sequence), 1)

        self.sequence.add_sequence(self.subsequence2)
        self.assertEqual(len(self.sequence), 2)

    def test_get_by_starting_point(self):
        """
        Test the get_by_starting_point method of the Sequence class

        The method should return the subsequence with the starting point t

        The subsequences are:
                * Subsequence 1: instances=[1, 2, 3, 4], date=2021-1-1, starting_point=0
                * Subsequence 2: instances=[5, 6, 7, 8], date=2021-1-2, starting_point=4

        The expected output is:
                * Subsequence 1 for starting point 0
                * Subsequence 2 for starting point 4
                * None for starting point 8
        """

        self.sequence.add_sequence(self.subsequence1)
        self.sequence.add_sequence(self.subsequence2)
        self.assertEqual(self.sequence.get_by_starting_point(0), self.subsequence1)
        self.assertEqual(self.sequence.get_by_starting_point(4), self.subsequence2)
        self.assertIsNone(self.sequence.get_by_starting_point(8))

    def test_get_starting_points(self):
        """
        Test the get_starting_points method of the Sequence class

        The method should return the starting points of the subsequences

        The subsequences are:
                * Subsequence 1: instances=[1, 2, 3, 4], date=2021-1-1, starting_point=0
                * Subsequence 2: instances=[5, 6, 7, 8], date=2021-1-2, starting_point=4

        The expected output is:
                * [0, 4]
        """

        self.sequence.add_sequence(self.subsequence1)
        self.sequence.add_sequence(self.subsequence2)
        self.assertEqual(self.sequence.get_starting_points(), [0, 4])

    def test_get_dates(self):
        """
        Test the get_dates method of the Sequence class

        The method should return the dates of the subsequences

        The subsequences are:

                * Subsequence 1: instances=[1, 2, 3, 4], date=2021-1-1, starting_point=0
                * Subsequence 2: instances=[5, 6, 7, 8], date=2021-1-2, starting_point=4

        The expected output is:
                * [2021-1-1, 2021-1-2]
        """
        self.sequence.add_sequence(self.subsequence1)
        self.sequence.add_sequence(self.subsequence2)
        self.assertEqual(self.sequence.get_dates(), [datetime.date(2021, 1, 1), datetime.date(2021, 1, 2)])

    def test_get_subsequences(self):
        """
        Test the get_subsequences method of the Sequence class

        The method should return the subsequences

        The subsequences are:
                * Subsequence 1: instances=[1, 2, 3, 4], date=2021-1-1, starting_point=0
                * Subsequence 2: instances=[5, 6, 7, 8], date=2021-1-2, starting_point=4

        The expected output is:
                * [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])]
        """

        expected_output = [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])]
        self.sequence.add_sequence(self.subsequence1)
        self.sequence.add_sequence(self.subsequence2)
        self.assertTrue(np.array_equal(self.sequence.get_subsequences(), expected_output))

    def test_extract_components(self):
        """
        Extract the components of a sequence

        This method should return a tuple of an array, a list of dates and a list of integers corresponding with the subsequences

        The sequence is:
            Sequence(
                list_sequences=[
                    Subsequence(instance=[1, 2, 3, 4], date=2021-1-1, starting_point=0),

                    Subsequence(instance=[5, 6, 7, 8], date=2021-1-2, starting_point=4)
                ]
            )

        The expected output is:
            If flatten = True:
                * (np.array([1, 2, 3, 4, 5, 6, 7, 8]), [datetime.date(2021, 1, 1), datetime.date(2021, 1, 2)], [0, 4])
            If flatten = False:
                *  (np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), [datetime.date(2021, 1, 1), datetime.date(2021, 1, 2)], [0, 4])
        """

        self.sequence.add_sequence(self.subsequence1)
        self.sequence.add_sequence(self.subsequence2)

        expected_output = (np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), [datetime.date(2021, 1, 1), datetime.date(2021, 1, 2)], [0, 4])
        expected_output_flatten = (np.array([1, 2, 3, 4, 5, 6, 7, 8]), [datetime.date(2021, 1, 1), datetime.date(2021, 1, 2)], [0, 4])

        flatten_subseq, flatten_dates, flatten_starting_points = self.sequence.extract_components(flatten=True)
        subseq, dates, starting_points = self.sequence.extract_components(flatten=False)

        # CASE 1: Flatten = True
        self.assertTrue(np.array_equal(flatten_subseq, expected_output_flatten[0]))
        self.assertEqual(flatten_dates, expected_output_flatten[1])
        self.assertTrue(np.array_equal(flatten_starting_points, expected_output_flatten[2]))

        # CASE 2: Flatten = False
        self.assertTrue(np.array_equal(subseq, expected_output[0]))
        self.assertEqual(dates, expected_output[1])
        self.assertEqual(starting_points, expected_output[2])

    def test_check_distinct_lengths(self):
        """
        The test should raise a ValueError if the subsequences have different lengths

        The subsequences are:
                * Subsequence 1: instances=[1, 2, 3, 4], date=2021-1-1, starting_point=0
                * Subsequence 2: instances=[5, 6, 7, 8], date=2021-1-2, starting_point=4
                * other length subsequence: instances=[1, 2, 3, 4, 5], date=2021-1-3, starting_point=8

        The expected output is:
                * ValueError
        """

        sequence = Sequence()
        sequence.add_sequence(self.subsequence1)
        sequence.add_sequence(self.subsequence2)

        with self.assertRaises(ValueError):
            sequence.add_sequence(self.other_length)


class TestCluster(unittest.TestCase):
    """
    Test the Cluster class

    The Cluster class is a class that represents a cluster of subsequences. It has the following methods:

        * __init__: initializes the Cluster object
        * add_instance: adds a subsequence to the cluster
        * get_sequences: returns the sequences of the cluster
        * update_centroid: updates the centroid of the cluster
        * get_starting_points: returns the starting points of the subsequences
        * get_dates: returns the dates of the subsequences
        * centroid_getter: returns the centroid of the cluster
        * centroid_setter: sets the centroid of the cluster


    """

    def setUp(self):
        self.sequence = Sequence()
        self.subsequence1 = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
        self.subsequence2 = Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)
        self.sequence.add_sequence(self.subsequence1)
        self.sequence.add_sequence(self.subsequence2)
        self.cluster = Cluster(centroid=np.array([10, 10, 10, 10]), instances=self.sequence)

    def test__eq__(self):
        """
        Test the __eq__ method of the Cluster class

        The method should return True if the two clusters are equal, False otherwise

        The other clusters are:
                * first: centroid=[10, 10, 10, 10], instances=[1, 2, 3, 4], date=2021-1-1, starting_point=0
                * second: centroid=[10, 10, 10, 10], instances=[5, 6, 7, 8], date=2021-1-2, starting_point=4
                * third: centroid=[10, 10, 10, 10], instances=[1, 2, 3, 4], date=2021-1-1, starting_point=0
                * fourth: centroid=[10, 10, 10, 10], instances=[1, 2, 3, 4], date=2021-1-1, starting_point=0

        The expected cluster is:
                * Cluster: centroid=[10, 10, 10, 10], instances=[1, 2, 3, 4], date=2021-1-1, starting_point=0

        The expected output is:
                * True for the first cluster
                * False for the second cluster
                * True for the third cluster
                * True for the fourth cluster
        """
        # CASE 1: True
        first = Cluster(np.array([10, 10, 10, 10]), self.sequence)
        self.assertTrue(self.cluster == first)

        # CASE 2: False
        second = Cluster(np.array([10, 10, 10, 10]), self.sequence)
        second.add_instance(self.subsequence2)
        self.assertFalse(self.cluster == second)

        # CASE 3: True
        third = Cluster(np.array([10, 10, 10, 10]), self.sequence)
        self.assertTrue(self.cluster == third)

        # CASE 4: True
        fourth = Cluster(np.array([10, 10, 10, 10]), self.sequence)
        self.assertTrue(self.cluster == fourth)

    def test_add_instance(self):
        """
        Test the add_instance method of the Cluster class

        The method should add a subsequence to the cluster

        The instance of the cluster is formed by the following Sequence:
            Sequence(
                list_sequences=[
                    Subsequence(instance=[1, 2, 3, 4], date=2021-1-1, starting_point=0),

                    Subsequence(instance=[5, 6, 7, 8], date=2021-1-2, starting_point=4)
                ]
            )

        The new subsequence to add is:
                * new subsequence: instances=[9, 10, 11, 12], date=2021-1-3, starting_point=8

        The expected output is:
                * 3 for the length of the cluster after adding the new subsequence
        """

        new_subsequence = Subsequence(np.array([9, 10, 11, 12]), datetime.date(2021, 1, 3), 8)
        self.cluster.add_instance(new_subsequence)
        self.assertEqual(len(self.cluster.get_sequences()), 3)

    def test_get_sequences(self):
        """

        Test the get_sequences method of the Cluster class

        The method should return the sequences of the cluster

        The instance of the cluster is formed by the following Sequence:

            Sequence(
                list_sequences=[
                    Subsequence(instance=[1, 2, 3, 4], date=2021-1-1, starting_point=0),

                    Subsequence(instance=[5, 6, 7, 8], date=2021-1-2, starting_point=4)
                ]
            )

        The expected output is:
                * [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])]
        """
        subsequences_cluster = self.cluster.get_sequences().get_subsequences()
        expected_output = [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])]

        self.assertTrue(np.array_equal(subsequences_cluster, expected_output))

    def test_update_centroid(self):
        """
        Test the update_centroid method of the Cluster class

        The method should update the centroid of the cluster

        The instance of the cluster is formed by the following Sequence:

        Cluster(
            centroid=[10, 10, 10, 10],

            instances= Sequence(
                list_sequences=[
                    Subsequence(instance=[1, 2, 3, 4], date=2021-1-1, starting_point=0),

                    Subsequence(instance=[5, 6, 7, 8], date=2021-1-2, starting_point=4)
                ]))

        The actual centroid is:
        [10, 10, 10, 10]

        The expected centroid after executing the update_centroid method is:
        [3.0, 4.0, 5.0, 6.0]
        """

        self.cluster.update_centroid()
        self.assertTrue(np.array_equal(self.cluster.centroid, np.array([3.0, 4.0, 5.0, 6.0])))

    def test_get_starting_points(self):
        """
        Test the get_starting_points method of the Cluster class

        The method should return the starting points of the subsequences

        The instance of the cluster is formed by the following Sequence:

        Cluster(
            centroid=[10, 10, 10, 10],

            instances= Sequence(
                list_sequences=[
                    Subsequence(instance=[1, 2, 3, 4], date=2021-1-1, starting_point=0),

                    Subsequence(instance=[5, 6, 7, 8], date=2021-1-2, starting_point=4)
                ]))

        The expected output is:
            * [0, 4]

        """

        self.assertEqual(self.cluster.get_starting_points(), [0, 4])

    def test_get_dates(self):
        """
        Test the get_dates method of the Cluster class

        The method should return the dates of the subsequences

        The cluster is formed by the following Sequence:

        Cluster(
            centroid=[10, 10, 10, 10],

            instances= Sequence(
                list_sequences=[
                    Subsequence(instance=[1, 2, 3, 4], date=2021-1-1, starting_point=0),

                    Subsequence(instance=[5, 6, 7, 8], date=2021-1-2, starting_point=4)
                ]))

        The expected output is:
            * [2021-1-1, 2021-1-2]
        """

        self.assertEqual(self.cluster.get_dates(), [datetime.date(2021, 1, 1), datetime.date(2021, 1, 2)])

    def test_centroid_getter(self):
        """
        Test the centroid_getter method of the Cluster class

        The method should return the centroid of the cluster

        The cluster is formed by the following Sequence:

        Cluster(
            centroid=[10, 10, 10, 10],

            instances= Sequence(
                list_sequences=[
                    Subsequence(instance=[1, 2, 3, 4], date=2021-1-1, starting_point=0),

                    Subsequence(instance=[5, 6, 7, 8], date=2021-1-2, starting_point=4)
                ]))

        The expected output is:
            * [10, 10, 10, 10]
        """

        self.assertTrue(np.array_equal(self.cluster.centroid, np.array([10, 10, 10, 10])))

    def test_centroid_setter(self):
        """
        Test the centroid_setter method of the Cluster class

        The method should set the centroid of the cluster

        The cluster is formed by the following Sequence:

        Cluster(
            centroid=[10, 10, 10, 10],

            instances= Sequence(
                list_sequences=[
                    Subsequence(instance=[1, 2, 3, 4], date=2021-1-1, starting_point=0),

                    Subsequence(instance=[5, 6, 7, 8], date=2021-1-2, starting_point=4)
                ]))

        The new centroid is:
        [1, 2, 3, 4]

        The expected centroid set is:
        [1, 2, 3, 4]
        """

        new_centroid = np.array([1, 2, 3, 4])
        self.cluster.centroid = new_centroid
        self.assertTrue(np.array_equal(self.cluster.centroid, new_centroid))

    def test_check_distinct_lengths(self):
        """
        The test should raise a ValueError if the instances have different length from the centroid
        or if a new added instance has a different length from the centroid

        The instances of the cluster are:
                * Subsequence 1: instances=[1, 2, 3, 4], date=2021-1-1, starting_point=0
                * Subsequence 2: instances=[5, 6, 7, 8], date=2021-1-2, starting_point=4

        The new centroid is:
                * [1, 3, 5, 7, 9]

        The new added instance is:
                * Subsequence 3: instances=[1, 2, 3, 4, 5], date=2021-1-3, starting_point=8

        The expected output for both cases is:
                * ValueError
        """

        # CASE 1: ValueError for the centroid
        with self.assertRaises(ValueError):
            Cluster(centroid=np.array([1, 3, 5, 7, 9]), instances=self.sequence)

        # CASE 2: ValueError for the new added instance
        with self.assertRaises(ValueError):
            self.cluster.add_instance(Subsequence(np.array([1, 2, 3, 4, 5]), datetime.date(2021, 1, 3), 8))


class TestRoutines(unittest.TestCase):
    """
    Test the Routines class

    The Routines class is a class that represents a set of clusters. It has the following methods:

        * __init__: initializes the Routines object
        * add_routine: adds a cluster to the routines
        * drop_indexes: drops the clusters with the indexes in the list
        * get_routines: returns the clusters
        * to_collection: returns the clusters in a collection

    """

    def setUp(self):
        """
        Set up the Routines object and the clusters for the tests

        The Routine is formed by the following cluster:

        Routines(
            list_routines=[
                Cluster(
                    - centroid = [3, 4, 5, 6],
                    - instances = [[1, 2, 3, 4], [5, 6, 7, 8]]
                    - starting_points = [0, 4]
                    - dates = [2021-1-1, 2021-1-2]
                )])
        """

        self.sequence = Sequence()
        self.subsequence1 = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
        self.subsequence2 = Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)
        self.sequence.add_sequence(self.subsequence1)
        self.sequence.add_sequence(self.subsequence2)
        self.cluster = Cluster(np.array([3, 4, 5, 6]), self.sequence)
        self.routines = Routines(self.cluster)

    def test_add_routine(self):
        """
        Test the add_routine method of the Routines class

        The method should add a cluster to the routines

        The cluster to add is:

        Cluster(
            - centroid = [7, 8, 9, 10],
            - instances = [[1, 2, 3, 4], [5, 6, 7, 8]]
            - starting_points = [0, 4]
            - dates = [2021-1-1, 2021-1-2]
        )

        The expected length of the routines after adding the cluster is:
            * 2
        """

        new_cluster = Cluster(np.array([7, 8, 9, 10]), self.sequence)
        self.routines.add_routine(new_cluster)
        self.assertEqual(len(self.routines), 2)

    def test_drop_indexes(self):
        """
        Test the drop_indexes method of the Routines class

        The method should return a new routine instance without the clusters with the indexes in the list

        The routines are:

        Routines(
            list_routines=[
                Cluster(
                    - centroid = [7, 8, 9, 10],
                    - instances = [[1, 2, 3, 4], [5, 6, 7, 8]]
                    - starting_points = [0, 4]
                    - dates = [2021-1-1, 2021-1-2]
                ),
                Cluster(
                    - centroid = [7, 8, 9, 10],
                    - instances = [[1, 2, 3, 4], [5, 6, 7, 8]]
                    - starting_points = [0, 4]
                    - dates = [2021-1-1, 2021-1-2]
                )])

        The actual length of the routines is:
                * 2

        The method is expected to drop the first cluster (index 0)

        The expected length of the routines after dropping the cluster is:
            * 1
        """

        self.routines.add_routine(self.cluster)
        self.routines = self.routines.drop_indexes([0])
        self.assertEqual(len(self.routines), 1)

    def test_get_routines(self):
        """
        Test the get_routines method of the Routines class

        The method should return a list of clusters

        The routines are:

        Routines(
            list_routines=[
                Cluster(
                    - centroid = [3, 4, 5, 6],
                    - instances = [[1, 2, 3, 4], [5, 6, 7, 8]]
                    - starting_points = [0, 4]
                    - dates = [2021-1-1, 2021-1-2]
                )])

        The expected output is:

        [Cluster(
            - centroid = [3, 4, 5, 6],
            - instances = [[1, 2, 3, 4], [5, 6, 7, 8]]
            - starting_points = [0, 4]
            - dates = [2021-1-1, 2021-1-2]
        )]
        """

        self.assertEqual(self.routines.get_routines(), [self.cluster])

    def test_to_collection(self):
        """
        Test the to_collection method of the Routines class

        The method should return the clusters in a collection

        The routines are:

        Routines(
            list_routines=[
                Cluster(
                    - centroid = [3, 4, 5, 6],
                    - instances = [[1, 2, 3, 4], [5, 6, 7, 8]]
                    - starting_points = [0, 4]
                    - dates = [2021-1-1, 2021-1-2]
                )])

        The expected output is:

        >>> [{'centroid': np.array([3, 4, 5, 6]),
        >>> 'instances': [
        >>>  ...  {'instance': [1, 2, 3, 4], 'date': datetime.date(2021, 1, 1), 'starting_point': 0},
        >>>  ...  {'instance': [5, 6, 7, 8], 'date': datetime.date(2021, 1, 2), 'starting_point': 4}
        >>> ] }]
        """

        collection = self.routines.to_collection()
        expected_collection = [{'centroid': np.array([3, 4, 5, 6]), 'instances': self.sequence.get_subsequences()}]
        self.assertEqual(np.array_equal(collection[0]["centroid"], expected_collection[0]["centroid"]), True)
        self.assertTrue(
            np.array_equal(collection[0]["instances"][0]["instance"], expected_collection[0]["instances"][0]))


class TestHierarchyRoutine(unittest.TestCase):
    def setUp(self):
        """
        Set up the HierarchyRoutine object and the routines for the tests

        The routines are:

        Routine1:

        Routines(
            list_routines=[
                Cluster(
                    - centroid = [2, 3, 4],
                    - instances = [[1, 2, 3], [4, 5, 6]]
                    - starting_points = [0, 3]
                    - dates = [2021-1-1, 2021-1-2]
                ),
                Cluster(
                    - centroid = [5, 6, 7],
                    - instances = [[7, 8, 9]]
                    - starting_points = [6]
                    - dates = [2021-1-3]
                )])

        Routine2:

        Routines(
            list_routines=[
                Cluster(
                    - centroid = [3, 4, 5, 6],
                    - instances = [[1, 2, 3, 4], [5, 6, 7, 8]]
                    - starting_points = [0, 4]
                    - dates = [2021-1-1, 2021-1-2]
                ),
                Cluster(
                    - centroid = [7, 8, 9, 10],
                    - instances = [[9, 10, 11, 12]]
                    - starting_points = [8]
                    - dates = [2021-1-3]
                )])

        Routine3:

        Routines(
            list_routines=[
                Cluster(
                    - centroid = [4, 5, 6, 7, 8],
                    - instances = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
                    - starting_points = [0, 5]
                    - dates = [2021-1-1, 2021-1-2]
                ),
                Cluster(
                    - centroid = [9, 10, 11, 12, 13],
                    - instances = [[11, 12, 13, 14, 15]]
                    - starting_points = [10]
                    - dates = [2021-1-3]
                )])
        """

        subseq1_3 = Subsequence(np.array([1, 2, 3]), datetime.date(2021, 1, 1), 0)
        subseq2_3 = Subsequence(np.array([4, 5, 6]), datetime.date(2021, 1, 2), 3)
        subseq3_3 = Subsequence(np.array([7, 8, 9]), datetime.date(2021, 1, 3), 6)

        subseq1_4 = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
        subseq2_4 = Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)
        subseq3_4 = Subsequence(np.array([9, 10, 11, 12]), datetime.date(2021, 1, 3), 8)

        subseq1_5 = Subsequence(np.array([1, 2, 3, 4, 5]), datetime.date(2021, 1, 1), 0)
        subseq2_5 = Subsequence(np.array([6, 7, 8, 9, 10]), datetime.date(2021, 1, 2), 5)
        subseq3_5 = Subsequence(np.array([11, 12, 13, 14, 15]), datetime.date(2021, 1, 3), 10)

        sequence1_3 = Sequence(subseq1_3)
        sequence1_3.add_sequence(subseq2_3)

        sequence2_3 = Sequence(subseq3_3)

        sequence1_4 = Sequence(subseq1_4)
        sequence1_4.add_sequence(subseq2_4)

        sequence2_4 = Sequence(subseq3_4)

        sequence1_5 = Sequence(subseq1_5)
        sequence1_5.add_sequence(subseq2_5)

        sequence2_5 = Sequence(subseq3_5)

        self.cluster1_3 = Cluster(np.array([2, 3, 4]), sequence1_3)
        self.cluster2_3 = Cluster(np.array([5, 6, 7]), sequence2_3)

        self.cluster1_4 = Cluster(np.array([3, 4, 5, 6]), sequence1_4)
        self.cluster2_4 = Cluster(np.array([7, 8, 9, 10]), sequence2_4)

        self.cluster1_5 = Cluster(np.array([4, 5, 6, 7, 8]), sequence1_5)
        self.cluster2_5 = Cluster(np.array([9, 10, 11, 12, 13]), sequence2_5)

        self.routines1 = Routines(self.cluster1_3)
        self.routines1.add_routine(self.cluster2_3)

        self.routines2 = Routines(self.cluster1_4)
        self.routines2.add_routine(self.cluster2_4)

        self.routines3 = Routines(self.cluster1_5)
        self.routines3.add_routine(self.cluster2_5)

        self.hierarchy_routine = HierarchyRoutine()

    def test_init(self):
        """
        Test the __init__ method of the HierarchyRoutine class
        """

        self.assertEqual(len(self.hierarchy_routine), 0)

    def test_setitem(self):
        """
        Test the __setitem__ method of the HierarchyRoutine class

        The method should add on the key (hierarchy) the value (routines)

        The routines are:
            routine1: hierarchy=3
            routine2: hierarchy=4
            routine3: hierarchy=5

        The expected hierarchy is:
            * {3: Routine1, 4: Routine2, 5: Routine3}

        Raises:
            TypeError: if the hierarchy is not an integer or the routine is not an instance of Routines
            ValueError: if the routine is empty or the hierarchy is not the same as the routine hierarchy
        """

        self.hierarchy_routine[3] = self.routines1
        self.assertEqual(self.hierarchy_routine._HierarchyRoutine__hierarchy[0], 3)

        self.hierarchy_routine[4] = self.routines2
        self.assertEqual(self.hierarchy_routine._HierarchyRoutine__hierarchy[1], 4)

        self.hierarchy_routine[5] = self.routines3
        self.assertEqual(self.hierarchy_routine._HierarchyRoutine__hierarchy[2], 5)

        # Case 1: TypeError
        with self.assertRaises(TypeError):
            self.hierarchy_routine["string"] = self.routines1

        with self.assertRaises(TypeError):
            self.hierarchy_routine[6] = "string"

        # Case 2: ValueError: routine is empty
        with self.assertRaises(ValueError):
            self.hierarchy_routine[6] = Routines()

        # Case 3: ValueError: hierarchy is not the same as the routine hierarchy
        with self.assertRaises(ValueError):
            self.hierarchy_routine[6] = self.routines1

    def test_getitem(self):
        """
        Test the __getitem__ method of the HierarchyRoutine class

        The method should return the routines of the hierarchy

        The routines are:
            routine1: hierarchy=3
            routine2: hierarchy=4
            routine3: hierarchy=5

        Raises:
            TypeError: if the hierarchy is not an integer
            KeyError: if the hierarchy is not found in the routines
        """

        self.hierarchy_routine[3] = self.routines1
        self.hierarchy_routine[4] = self.routines2
        self.hierarchy_routine[5] = self.routines3

        # Returns the routine
        self.assertEqual(self.hierarchy_routine[3], self.routines1)
        self.assertEqual(self.hierarchy_routine[4], self.routines2)
        self.assertEqual(self.hierarchy_routine[5], self.routines3)

        # Case 1: TypeError
        with self.assertRaises(TypeError):
            self.hierarchy_routine["string"]

        # Case 2: KeyError
        with self.assertRaises(KeyError):
            self.hierarchy_routine[6]

    def test_len(self):
        """
        Test the __len__ method of the HierarchyRoutine class

        The method should return the number of routines in the hierarchy

        The routines are:
            routine1: hierarchy=3
            routine2: hierarchy=4
            routine3: hierarchy=5

        The expected outputs are:
            * 0 (no routines added)
            * 1 (1 routines added)
            * 2 (2 routines added)
            * 3 (3 routines added)
        """

        self.assertEqual(len(self.hierarchy_routine), 0)

        self.hierarchy_routine[3] = self.routines1
        self.assertEqual(len(self.hierarchy_routine), 1)

        self.hierarchy_routine[4] = self.routines2
        self.assertEqual(len(self.hierarchy_routine), 2)

        self.hierarchy_routine[5] = self.routines3
        self.assertEqual(len(self.hierarchy_routine), 3)

    def test_contains(self):
        """
        Test the __contains__ method of the HierarchyRoutine class

        This method should check if the routine exists in the hierarchical routines

        The routines added in the hierarchy are:
            routine1: hierarchy=3
            routine2: hierarchy=4

        The routine not added in the hierarchy is:
            routine3: hierarchy=5

        The expected outputs are:
            * True (routine1 exists)
            * True (routine2 exists)
            * False (routine3 does not exist)

        Raises:
            TypeError: if the routine is not an instance of Routines
        """

        self.hierarchy_routine[3] = self.routines1
        self.hierarchy_routine[4] = self.routines2

        # Case 1: True
        self.assertTrue(self.routines1 in self.hierarchy_routine)

        # Case 2: True
        self.assertTrue(self.routines2 in self.hierarchy_routine)

        # Case 3: False
        self.assertFalse(self.routines3 in self.hierarchy_routine)

        # Case 4: TypeError
        with self.assertRaises(TypeError):
            "string" in self.hierarchy_routine

    def test_add_routine(self):
        """
        Test the add_routine method of the HierarchyRoutine class

        This method should add a routine to the HierarchyRoutine object.
        If the key (hierarchy) already exists, the routine is updated.
        Otherwise, the routine is added to the hierarchy.

        The routines added in the hierarchy are:
            routine1: hierarchy=3
            routine2: hierarchy=4

        The routine to add is:
            routine3: hierarchy=5

        The routine to update the routine1 is:
        Routine(
            list_clusters=[
                Cluster(
                    - centroid = [15, 16, 17]
                    - instances = [[13, 14, 15], [16, 17, 18]]
                    - starting_points = [12, 15]
                    - dates = [2021-1-4, 2021-1-5]
                ),
                Cluster(
                    - centroid = [18, 19, 20]
                    - instances = [[19, 20, 21]]
                    - starting_points = [18]
                    - dates = [2021-1-6]
                )])
            ]
        )

        Raises:
            TypeError: if the routine is not an instance of Routines
            ValueError: if the routine is empty
        """

        # Creating the routine to update the routine of hierarchy 3
        routine_to_update = Routines()
        subseq1 = Subsequence(np.array([13, 14, 15]), datetime.date(2021, 1, 4), 12)
        subseq2 = Subsequence(np.array([16, 17, 18]), datetime.date(2021, 1, 5), 15)
        subseq3 = Subsequence(np.array([19, 20, 21]), datetime.date(2021, 1, 6), 18)
        sequence = Sequence(subseq1)
        sequence.add_sequence(subseq2)
        sequence2 = Sequence(subseq3)
        cluster1 = Cluster(np.array([15, 16, 17]), sequence)
        cluster2 = Cluster(np.array([18, 19, 20]), sequence2)
        routine_to_update.add_routine(cluster1)
        routine_to_update.add_routine(cluster2)

        # Case 1: Add a new routine
        self.hierarchy_routine.add_routine(self.routines1)

        # Check if the routine was added with the correct hierarchy
        self.assertEqual(self.hierarchy_routine[3], self.routines1)
        self.assertEqual(self.hierarchy_routine._HierarchyRoutine__hierarchy, [3])

        # Case 2: Add a new routine
        self.hierarchy_routine.add_routine(self.routines2)

        # Check if the routine was added with the correct hierarchy
        self.assertEqual(self.hierarchy_routine[4], self.routines2)
        self.assertEqual(self.hierarchy_routine._HierarchyRoutine__hierarchy, [3, 4])

        # Case 3: Add a new routine
        self.hierarchy_routine.add_routine(self.routines3)

        # Check if the routine was added with the correct hierarchy
        self.assertEqual(self.hierarchy_routine[5], self.routines3)
        self.assertEqual(self.hierarchy_routine._HierarchyRoutine__hierarchy, [3, 4, 5])

        # Case 4: Update the routine of hierarchy 3
        self.hierarchy_routine.add_routine(routine_to_update)

        # Check if the routine was updated with the correct hierarchy
        self.assertEqual(self.hierarchy_routine[3], routine_to_update)

        # Case 5: TypeError: routine is not an instance of Routines
        with self.assertRaises(TypeError):
            self.hierarchy_routine.add_routine("string")

        # Case 6: ValueError: routine is empty
        with self.assertRaises(ValueError):
            self.hierarchy_routine.add_routine(Routines())

    def test_keys(self):
        """
        Test the keys property of the HierarchyRoutine class

        This property is a getter that returns the hierarchy of the routines as list of integers

        The routines added in the hierarchy are:
            routine1: hierarchy=3
            routine2: hierarchy=4
            routine3: hierarchy=5

        The expected output is:
            * [3, 4, 5]
        """

        self.hierarchy_routine.add_routine(self.routines1)
        self.hierarchy_routine.add_routine(self.routines2)
        self.hierarchy_routine.add_routine(self.routines3)

        self.assertEqual(self.hierarchy_routine.keys, [3, 4, 5])

    def test_values(self):
        """
        Test the values property of the HierarchyRoutine class

        This property is a getter that returns the routines as list of Routines

        The routines added in the hierarchy are:
            routine1: hierarchy=3
            routine2: hierarchy=4
            routine3: hierarchy=5

        The expected output is:
            * [Routine1, Routine2, Routine3]
        """

        self.hierarchy_routine.add_routine(self.routines1)
        self.hierarchy_routine.add_routine(self.routines2)
        self.hierarchy_routine.add_routine(self.routines3)

        # Check if the values are correct
        self.assertEqual(self.hierarchy_routine.values, [self.routines1, self.routines2, self.routines3])

    def test_items(self):
        """
        Test the items property of the HierarchyRoutine class.

        This method should return an iterator of each tuple (hierarchy, routine) in the HierarchyRoutine object.

        The routines added in the hierarchy are:
            routine1: hierarchy=3
            routine2: hierarchy=4
            routine3: hierarchy=5

        The expected output is:
            * [(3, Routine1), (4, Routine2), (5, Routine3)]
        """

        self.hierarchy_routine.add_routine(self.routines1)
        self.hierarchy_routine.add_routine(self.routines2)
        self.hierarchy_routine.add_routine(self.routines3)

        # Check if the items are correct
        for key, value in self.hierarchy_routine.items:
            self.assertEqual(self.hierarchy_routine[key], value)

    def test_to_dictionary(self):
        """
        Test the to_dictionary method of the HierarchyRoutine class

        The method should return the routines in a dictionary

        The routines added in the hierarchy are:
            routine1: hierarchy=3
            routine2: hierarchy=4
            routine3: hierarchy=5

        The expected output is:
            * {3: Routine1, 4: Routine2, 5: Routine3}
        """

        self.hierarchy_routine.add_routine(self.routines1)

        expected_output = {3: [{'centroid': np.array([2, 3, 4]),
                                'instances': [{'instance': np.array([1, 2, 3]), 'date': datetime.date(2021, 1, 1),
                                               'starting_point': 0},
                                              {'instance': np.array([4, 5, 6]), 'date': datetime.date(2021, 1, 2),
                                               'starting_point': 3}]},
                               {'centroid': np.array([5, 6, 7]),
                                'instances': [{'instance': np.array([7, 8, 9]), 'date': datetime.date(2021, 1, 3),
                                               'starting_point': 6}]}]
                           }

        # Check if the dictionary is correct
        estimated_output = self.hierarchy_routine.to_dictionary()

        #
        for key, value in estimated_output.items():
            for i, cluster in enumerate(value):
                # Check if the centroids match
                self.assertTrue(np.array_equal(cluster["centroid"], expected_output[key][i]["centroid"]))

                # Check if the instances match
                for j, instance in enumerate(cluster["instances"]):
                    self.assertTrue(
                        np.array_equal(instance["instance"], expected_output[key][i]["instances"][j]["instance"]))
                    self.assertTrue(np.array_equal(instance["date"], expected_output[key][i]["instances"][j]["date"]))
                    self.assertTrue(np.array_equal(instance["starting_point"],
                                                   expected_output[key][i]["instances"][j]["starting_point"]))


class TestDRFL(unittest.TestCase):
    """
    Test the DRFL class

    The DRFL class is a class that implements the DRFL algorithm. The DRFL algorithm is a clustering algorithm that

    The DRFL class has the following methods:
        * __minimum_distance_index: returns the index of the minimum distance in the list or array
        * _extract_subsequence: extracts a subsequence from the time series
        * __is_match: returns True if the distance between the subsequences is less than _epsilon, False otherwise
        * _subgroup: returns a Routines object with the clusters obtained from the time series
    """

    def setUp(self):
        """
        Set up the DRFL object and the time series for the tests

        The parameters of the DRFL object are:
            * _m = 3
            * _G = 4
            * _R = 2
            * _C = 3
            * _epsilon = 1

        The time series is:
        [1, 3, 6, 4, 2, 1, 2, 3, 6, 4, 1, 1, 3, 6, 4, 1]
        """
        self._m = 3
        self._G = 4
        self._R = 2
        self._C = 3
        self._epsilon = 1
        self.time_series = pd.Series([1, 3, 6, 4, 2, 1, 2, 3, 6, 4, 1, 1, 3, 6, 4, 1])
        self.time_series.index = pd.date_range(start="2024-01-01", periods=len(self.time_series))
        self.drfl = DRFL(m=3, G=4, R=2, C=3, epsilon=1)
        self.drfl_fitted = DRFL(m=3, G=4, R=2, C=3, epsilon=1)
        self.drfl_fitted.fit(self.time_series)

    def test__minimum_distance_index(self):
        """
        Test the __minimum_distance_index method of the DRFL class

        The method should return the index of the minimum distance in the list or array

        The list and the array have the following values:
        [1, 2, 0, 4]

        The expected output is:
        * 2

        There should be an expected type error for the following cases:
            * distances is not an index or an array
        """
        # Case 1: list as input
        distances = [1, 2, 0, 4]
        self.assertEqual(self.drfl._DRFL__minimum_distance_index(distances), 2)

        # Case 2: np.array as input
        distances = np.array([1, 2, 0, 4])
        self.assertEqual(self.drfl._DRFL__minimum_distance_index(distances), 2)

        # Case 3: Not an index or an array
        distances = "Not an index"
        with self.assertRaises(TypeError):
            self.drfl._DRFL__minimum_distance_index(distances)

    def test__extract_subsequence(self):
        """
        Test the _extract_subsequence method of the DRFL class

        The method should extract a subsequence from the time series

        The time series is:
        [1, 3, 6, 4, 2, 1, 2, 3, 6, 4, 1, 1, 3, 6, 4, 1]

        The expected output is:
            * Subsequence: [1, 3, 6], Date: 2024-1-1, Starting Point: 0

        There should be an expected type error for the following case:
            * t is not an integer

        There should be an expected value error for the following case:
            * t is out of the range of the time series
        """

        # Case 1: extracts correctly the first subsequence
        self.drfl._extract_subsequence(self.time_series, 0)
        sequence = self.drfl._DRFL__sequence.get_by_starting_point(0)
        self.assertEqual(sequence.get_instance().tolist(), [1, 3, 6])
        self.assertEqual(sequence.get_date(), datetime.date(2024, 1, 1))
        self.assertEqual(sequence.get_starting_point(), 0)

        # Case 2: check if t is an integer
        with self.assertRaises(TypeError):
            self.drfl._extract_subsequence(self.time_series, 0.5)
            self.drfl._extract_subsequence(self.time_series, "not a integer")

        # Case 3: check if t is in the range of the time series
        with self.assertRaises(ValueError):
            self.drfl._extract_subsequence(self.time_series, -1)
            self.drfl._extract_subsequence(self.time_series, 100)

    def test__IsMatch(self):
        """
        Test the __is_match method of the DRFL class

        The method should return True if the distance between the subsequences is less than _epsilon, False otherwise

        The time series is:
        [1, 3, 6, 4, 2, 1, 2, 3, 6, 4, 1, 1, 3, 6, 4, 1]

        The parameters of the DRFL object are:
              * _R = 2

        The subsequences are:
            * Subsequence1: [1, 2, 3], Date: 2024-1-1, Starting Point: 0
            * Subsequence2: [3, 4, 5], Date: 2024-1-2, Starting Point: 0
            * Subsequence3: [3, 4, 6], Date: 2024-1-2, Starting Point: 0

        The expected output is:
            * True for Subsequence1 and Subsequence2
            * False for Subsequence1 and Subsequence3

        There should be an expected type error for the following cases:
            * S1 is not a Subsequence
            * S2 is not a Subsequence or an array
        """

        subsequence1 = Subsequence(np.array([1, 2, 3]), datetime.date(2024, 1, 1), 0)
        subsequence2 = Subsequence(np.array([3, 4, 5]), datetime.date(2024, 1, 2), 0)
        subsequence3 = Subsequence(np.array([3, 4, 6]), datetime.date(2024, 1, 2), 0)

        # Case 1: S1 Subsequence; S2: Subsequence
        self.assertTrue(self.drfl._DRFL__is_match(S1=subsequence1, S2=subsequence2, R=self._R))
        self.assertFalse(self.drfl._DRFL__is_match(S1=subsequence1, S2=subsequence3, R=self._R))

        # Case 2: S1 Subsequence; S2: Array
        self.assertTrue(self.drfl._DRFL__is_match(S1=subsequence1, S2=np.array([1, 2, 3]), R=self._R))
        self.assertFalse(self.drfl._DRFL__is_match(S1=subsequence1, S2=np.array([1, 2, 6]), R=self._R))

        # Case 3: S1 other type than Subsequence; S2: Subsequence
        with self.assertRaises(TypeError):
            self.drfl._DRFL__is_match(S1=np.array([1, 2, 3]), S2=subsequence1, R=self._R)
            self.drfl._DRFL__is_match(S1="Not a Subsequence", S2=subsequence1, R=self._R)

        # Case 4: S1 Subsequence; S2: other type than array an instance
        with self.assertRaises(TypeError):
            self.drfl._DRFL__is_match(S1=subsequence1, S2="Not an array", R=self._R)
            self.drfl._DRFL__is_match(S1=subsequence1, S2=[1, 2, 3], R=self._R)

    def test__SubGroup(self):
        """
        Test the _subgroup method of the DRFL class

        The method should return a Routines object with the clusters obtained from the time series

        The time series is:
        [1, 3, 6, 4, 2, 1, 2, 3, 6, 4, 1, 1, 3, 6, 4, 1]

        The parameters of the DRFL object are:
           * _m = 3
           * _G = 4
           * _R = 2
           * _C = 3
           * _epsilon = 1

        The expected output is:

        Cluster 1:
            Centroid: [4/3, 3.0, 6.0]

            Instances:
                - Subsequence: [1, 3, 6], Date: 2024-1-1, Starting Point: 0
                - Subsequence: [2, 3, 6], Date: 2024-1-7, Starting Point: 6
                - Subsequence: [1, 3, 6], Date: 2024-1-12, Starting Point: 11

        Cluster 2:
            Centroid: [3.0, 6.0, 4.0]

            Instances:
                - Subsequence: [3, 6, 4], Date: 2024-1-2, Starting Point: 1
                - Subsequence: [3, 6, 4], Date: 2024-1-8, Starting Point: 7
                - Subsequence: [3, 6, 4], Date: 2024-1-13, Starting Point: 12

        Cluster 3:
            Centroid: [5.5, 3.5, 1.25]

            Instances:
                - Subsequence: [6, 4, 2], Date: 2024-1-3, Starting Point: 2
                - Subsequence: [4, 2, 1], Date: 2024-1-4, Starting Point: 3
                - Subsequence: [6, 4, 1], Date: 2024-1-9, Starting Point: 8
                - Subsequence: [6, 4, 1], Date: 2024-1-14, Starting Point: 13
        """

        # Expected instances for the first cluster
        expected_instances_centroid1 = Sequence(Subsequence(np.array([1, 3, 6]), pd.to_datetime("2024-1-1"), 0))
        expected_instances_centroid1.add_sequence(Subsequence(np.array([2, 3, 6]), pd.to_datetime("2024-1-7"), 6))
        expected_instances_centroid1.add_sequence(Subsequence(np.array([1, 3, 6]), pd.to_datetime("2024, 1, 12"), 11))

        # Expected instances for the second cluster
        expected_instances_centroid2 = Sequence(Subsequence(np.array([3, 6, 4]), pd.to_datetime("2024-1-2"), 1))
        expected_instances_centroid2.add_sequence(Subsequence(np.array([3, 6, 4]), pd.to_datetime("2024-1-8"), 7))
        expected_instances_centroid2.add_sequence(Subsequence(np.array([3, 6, 4]), pd.to_datetime("2024-1-13"), 12))

        # Expected instances for the third cluster
        expected_instances_centroid3 = Sequence(Subsequence(np.array([6, 4, 2]), pd.to_datetime("2024-1-3"), 2))
        expected_instances_centroid3.add_sequence(Subsequence(np.array([4, 2, 1]), pd.to_datetime("2024-1-4"), 3))
        expected_instances_centroid3.add_sequence(Subsequence(np.array([6, 4, 1]), pd.to_datetime("2024-1-9"), 8))
        expected_instances_centroid3.add_sequence(Subsequence(np.array([6, 4, 1]), pd.to_datetime("2024-1-14"), 13))

        expected_routine = Routines(Cluster(np.array([4 / 3, 3.0, 6.0]), expected_instances_centroid1))
        expected_routine.add_routine(Cluster(np.array([3.0, 6.0, 4.0]), expected_instances_centroid2))
        expected_routine.add_routine(Cluster(np.array([5.5, 3.5, 1.25]), expected_instances_centroid3))

        # Check if the routine is the expected
        routines_obtained_1 = self.drfl_fitted._subgroup(self.drfl_fitted._DRFL__sequence, self._R, self._C, self._G)
        routines_obtained_2 = self.drfl_fitted._subgroup(self.drfl_fitted._DRFL__sequence, R=3, C=2, G=2)

        self.assertEqual(routines_obtained_1, expected_routine)
        self.assertNotEquals(routines_obtained_2, expected_routine)


class TestDRGS(unittest.TestCase):
    def setUp(self):
        """
        Set up the DRGS object and the time series for the tests

        The time series is:
        [1, 3, 6, 4, 2, 1, 2, 3, 6, 4, 1, 1, 3, 6, 4, 1]
        """
        # Setup for a example sequence
        subsequence = Subsequence(np.array([1, 3, 6]), datetime.date(2024, 1, 1), 0)
        subsequence2 = Subsequence(np.array([2, 3, 6]), datetime.date(2024, 1, 7), 6)
        subsequence3 = Subsequence(np.array([1, 3, 6]), datetime.date(2024, 1, 12), 11)

        self.sequence = Sequence(subsequence)
        self.sequence.add_sequence(subsequence2)
        self.sequence.add_sequence(subsequence3)

        # Setup for the input time series
        self.time_series = pd.Series([1, 3, 6, 4, 2, 1, 2, 3, 6, 4, 1, 1, 3, 6, 4, 1])
        self.time_series.index = pd.date_range(start="2024-01-01", periods=len(self.time_series))

        # Setup for the DRGS object
        self.length_range = (3, 8)
        self.R = 2
        self.C = 3
        self.G = 4
        self.epsilon = 1
        self.L = 3
        self.drgs = DRGS(length_range=self.length_range, R=self.R, C=self.C, G=self.G, epsilon=self.epsilon, L=self.L)

        self.drgs_fitted = DRGS(length_range=self.length_range, R=self.R, C=self.C, G=self.G, epsilon=self.epsilon,
                                L=self.L)
        self.drgs_fitted.fit(self.time_series)

    def test__union_routines(self):
        """
        Test the __union_routines method of the DRGS class

        The method should return the union of the routines appending the clusters of the second routine to the first routine

        The routines are:

        left:
        Routines(
                list_routines=[
                    Cluster(
                        - centroid=np.array([1, 2, 3])
                        - instances=[[1, 2, 3]]
                        - date=datetime.date(2024, 1, 1),
                        - starting_point=[0]
                        )
                ])

        right:
        Routines(
                list_routines=[
                    Cluster(
                        - centroid=np.array([3, 2, 1])
                        - instances=[[3, 2, 1]]
                        - date=datetime.date(2024, 1, 1),
                        - starting_point=[0]
                        )
                ])

        The expected output is:
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
                ])

        Raises:
            TypeError: if the parameter is not an instance of Routines
            ValueError: if the routine is if the hierarchy of the routines is not the same
        """
        # Create the routines
        left_cluster = Cluster(centroid=np.array([1, 2, 3]), instances=Sequence(
            Subsequence(np.array([1, 2, 3]), date=datetime.date(2024, 1, 1), starting_point=0)))
        right_cluster = Cluster(centroid=np.array([3, 2, 1]), instances=Sequence(
            Subsequence(np.array([3, 2, 1]), date=datetime.date(2024, 1, 1), starting_point=0)))
        other_hierarchy = Cluster(centroid=np.array([1, 2, 3, 4]), instances=Sequence(
            Subsequence(np.array([1, 2, 3, 4]), date=datetime.date(2024, 1, 1), starting_point=0)))

        # expected join routine
        join = Routines(left_cluster)
        join.add_routine(right_cluster)

        # Input routines
        left = Routines(left_cluster)
        right = Routines(right_cluster)
        other_hierarchy_routine = Routines(other_hierarchy)

        # Case 0: union correct
        self.assertEqual(self.drgs._DRGS__union_routines(left, right), join)

        # Case 1: TypeError
        with self.assertRaises(TypeError):
            self.drgs._DRGS__union_routines(left, "string")

        # Case 2: left is an empty routine
        self.assertEqual(self.drgs._DRGS__union_routines(Routines(), right), right)

        # Case 3: right is an empty routine
        self.assertEqual(self.drgs._DRGS__union_routines(left, Routines()), left)

        # Case 4: ValueError
        with self.assertRaises(ValueError):
            self.drgs._DRGS__union_routines(left, other_hierarchy_routine)

    def test__grow_from_left(self):
        """
        Test the __grow_from_left method of the DRGS class

        This method should grow a sequence from the left side of the time series taking as input a Sequence

        The sequence is:
        Sequence(
            list_subsequences=[
                Subsequence(
                    - instance = [1, 3, 6]
                    - date = 2024-1-1
                    - starting_point = 0
                ),
                Subsequence(
                    - instance = [2, 3, 6]
                    - date = 2024-1-7
                    - starting_point = 6
                ),
                Subsequence(
                    - instance = [1, 3, 6]
                    - date = 2024-1-12
                    - starting_point = 11
                )
            ]
        )

        The expected output is:

        left:
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

        # Expected output
        expected_output = Sequence(Subsequence(np.array([1, 3, 6, 4]), datetime.date(2024, 1, 1), 0))
        expected_output.add_sequence(Subsequence(np.array([2, 3, 6, 4]), datetime.date(2024, 1, 7), 6))
        expected_output.add_sequence(Subsequence(np.array([1, 3, 6, 4]), datetime.date(2024, 1, 12), 11))

        # Check if the sequence is the expected
        self.assertEqual(self.drgs_fitted._DRGS__grow_from_left(self.sequence), expected_output)


if __name__ == '__main__':
    unittest.main()
