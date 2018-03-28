#! /usr/bin/python

import sys


def __euclidean_distance(pa, pb):
    distance = 0

    for idx, position in enumerate(pa):
        distance += ((position - pb[idx])**2)

    return distance**0.5


def knn(instances, test_instances, k):
    """
    Calculates the k nearest neighbor's and predicts the class of the new test_instances (tries)

    :param list instances: A list of tuples like [((<attributes>), <classification>), ...]
    :param list test_instances: The testing instances, composed of a list of attribute tuples like [(<attributes>), ...]
    :param integer k: The k factor for the algorithm
    :return: A list with the classification related to the test instances
    :rtype: list
    """

    classified = []

    for test_instance in test_instances:
        distances = [sys.maxsize for i in range(0, k)]
        distances_idx = [0 for i in range(0, k)]

        for idx_instance, instance in enumerate(instances):
            instance_distance = __euclidean_distance(instance[0], test_instance)

            for idx_distance, distance in enumerate(distances):
                if distance > instance_distance:
                    distances.insert(idx_distance, instance_distance)
                    distances_idx.insert(idx_distance, idx_instance)
                    distances.pop()
                    distances_idx.pop()

                    break

        knn_classes = [instances[i][1] for i in distances_idx[0:k]]

        counters = {key: knn_classes.count(key) for key in knn_classes}

        winner_counter = 0
        winner = 0

        for key in counters:
            if counters[key] > winner_counter:
                winner_counter = counters[key]
                winner = key

        classified.append((test_instance, winner))

    return classified
