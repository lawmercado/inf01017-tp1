#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
import csv
import logging
import argparse
import random

from data.handler import DataHandler
from ml.supervised.evaluation import decision_tree_kcrossvalidation, random_forest_kcrossvalidation, get_statistics


def setup_logger():

    class MyFilter(object):
        def __init__(self, level):
            self.__level = level

        def filter(self, log_record):
            return log_record.levelno <= self.__level

    main_logger = logging.getLogger("main")

    formatter = logging.Formatter("%(levelname)s: %(message)s")

    handler = logging.FileHandler("output.log", mode="w")
    handler.setLevel(logging.DEBUG)
    handler.filter(MyFilter(logging.DEBUG))
    handler.setFormatter(formatter)
    main_logger.addHandler(handler)

    main_logger.setLevel(logging.INFO)

    return main_logger


if __name__ == '__main__':
    logger = setup_logger()

    supported_data_sets = ["benchmark", "diabetes", "wine", "ionosphere", "cancer"]
    supported_algorithms = ["id3_decision_tree", "id3_random_forest"]
    supported_discretizations = ["mean", "information_gain", "quartiles"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", help="enables debugging", action="store_true")
    parser.add_argument("--data_set", type=str, help="the data set to test. Options are " + str(supported_data_sets))
    parser.add_argument("--algorithm", type=str, help="the algorithm to use. Options are " + str(supported_algorithms))
    parser.add_argument("--seed", type=int, help="the seed to consider in random numbers generation")
    parser.add_argument("--ntree", type=int, default=10, help="how many trees to generate. Defaults to 10")
    parser.add_argument("--discretization", type=str, default="mean", help="the method to use in discretization. Options are " + str(supported_discretizations))

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if args.data_set is not None:
        if args.data_set in supported_data_sets:
            filename = ""
            delimiter = ""
            class_attr = ""
            id_attr = None

            if args.data_set.strip() == "benchmark":
                filename = "sets/benchmark.csv"
                delimiter = ";"
                class_attr = "Joga"

            elif args.data_set.strip() == "diabetes":
                filename = "sets/diabetes.csv"
                delimiter = ","
                class_attr = "Outcome"

            elif args.data_set.strip() == "wine":
                filename = "sets/wine.csv"
                delimiter = ","
                class_attr = "Type"

            elif args.data_set.strip() == "ionosphere":
                filename = "sets/ionosphere.csv"
                delimiter = ","
                class_attr = "radar"

            elif args.data_set.strip() == "cancer":
                filename = "sets/cancer.csv"
                delimiter = ","
                class_attr = "diagnosis"
                id_attr = "id"

            rows = list(csv.reader(open(filename, "r"), delimiter=delimiter))
            data_handler = DataHandler(rows, class_attr, id_attr)

            print("Discretizing...")

            if args.discretization == "mean":
                data_handler = data_handler.discretize()
            elif args.discretization == "quartile":
                data_handler = data_handler.discretize_quartile()
            elif args.discretization == "information_gain":
                data_handler = data_handler.discretize_information_gain()

            print("Processing...")

            if args.algorithm in supported_algorithms:
                if args.algorithm == "id3_random_forest":
                    print(get_statistics(random_forest_kcrossvalidation(data_handler, 10, args.ntree)))

                elif args.algorithm == "id3_decision_tree":
                    logger.info(get_statistics(decision_tree_kcrossvalidation(data_handler, 10)))

            print("See the log output is in output.log")

        else:
            raise AttributeError("Data set is not supported!")

    else:
        print("Nothing to do here...")
