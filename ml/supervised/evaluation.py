#! /usr/bin/python

from __future__ import division
from __future__ import print_function
from ml.supervised.algorithms import knn_classification, random_trees_classification


def knn_kcrossvalidation(data_handler, knn_factor, k_folds):
    folds = data_handler.stratify(k_folds)

    measures = {"acc": [], "f-measure": []}

    for idx_fold, fold in enumerate(folds):
        aux_folds = list(folds)  # Copy the folds
        test_instances = [instance[0] for instance in aux_folds.pop(idx_fold)]

        train_instances = []
        for aux_fold in aux_folds:
            train_instances += aux_fold

        classified_instances = knn_classification(train_instances, test_instances, knn_factor)

        # Compare classified instances with the test set
        correct_classifications = 0
        true_positive = 0
        false_positive = 0
        false_negative = 0
        true_negative = 0

        for (predicted_instance, test_instance) in zip(classified_instances, fold):
            if predicted_instance[1] == test_instance[1]:
                correct_classifications += 1
                if predicted_instance[1] == 1:
                    true_positive += 1
                else:
                    true_negative += 1

            elif predicted_instance[1] == 1:
                false_positive += 1

            elif predicted_instance[1] == 0:
                false_negative += 1

        # Generate the statistics
        acc = correct_classifications / len(classified_instances)
        measures["acc"].append(acc)

        rev = true_positive / (true_positive + false_negative)

        prec = true_positive / (true_positive + false_positive)

        f_measure = 2 * (prec * rev) / (prec + rev)
        measures["f-measure"].append(f_measure)

    return measures


def knn_repeatedkcrossvalidation(data_transformer, knn_factor, k_folds, repetitions):
    measures = {"acc": [], "f-measure": []}

    for i in range(0, repetitions):
        fold_measures = knn_kcrossvalidation(data_transformer, knn_factor, k_folds)

        measures["acc"] += fold_measures["acc"]
        measures["f-measure"] += fold_measures["f-measure"]

    return measures


def random_trees_kcrossvalidation(data_handler, k_folds, classes):
    """
    :param data_handler: Raw data for the cross validation
    :param k_folds: Number of folds to generate
    :param classes: List with possible class values
    :return: Tuple with values for accuracy and the F-measure
    """
    folds = data_handler.in_folds(k_folds)

    measures = {"acc": [], "f-measure": []}

    for index_fold, fold in enumerate(folds):
        aux_folds = list(folds)  # Copy the folds
        test_fold = aux_folds.pop(index_fold)

        test_handler = data_handler.fold_handler(test_fold)
        train_handler = data_handler.fold_handler(aux_folds)

        # Train the algorithm & Classify the test fold

        # classified_samples = random_trees(train_fold_handler, test_fold_handler)
        classified_samples = []

        # Initialize confusion matrix
        correct_classifications = []
        true_positives = []
        true_negatives = []
        false_positives = []
        false_negatives = []
        for i in range(len(classes)):
            correct_classifications[i] = 0
            true_positives[i] = 0
            true_negatives[i] = 0
            false_positives[i] = 0
            false_negatives[i] = 0

        # Compare classified samples with the test set
        for (predicted_sample, test_sample) in zip(classified_samples, fold):
            for class_index in range(len(classes)):
                if predicted_sample[class_index] == test_sample[class_index]:
                    correct_classifications[class_index] += 1
                    if predicted_sample[class_index] == classes[class_index]:
                        true_positives[class_index] += 1
                    else:
                        true_negatives[class_index] += 1
                elif predicted_sample[class_index] == classes[class_index]:
                    false_positives[class_index] += 1
                else:
                    false_negatives[class_index] += 1

        # Generate the statistics for each class
        acc = 0
        total_true_positives = 0
        total_false_positives = 0
        total_false_negatives = 0
        for class_index in range(len(classes)):
            acc += correct_classifications[class_index]

            total_true_positives += true_positives[class_index]
            total_false_positives += false_positives[class_index]
            total_false_negatives += false_negatives[class_index]

        acc = acc / len(classified_samples)
        measures["acc"].append(acc)

        rev_micro = total_true_positives / (total_true_positives + total_false_negatives)
        prec_micro = total_true_positives / (total_true_positives + total_false_positives)

        f_measure = 2 * (prec_micro * rev_micro) / (prec_micro + rev_micro)
        measures["f-measure"].append(f_measure)

    return measures


def get_statistics(measures):
    """
    With a set of measures, calculates the average and de standard

    :param dict measures: The name of the measures and a list of measurement
    :return: A tuple containing the average and the standard deviation associated with the measure
    :rtype: dict { measure: (<average>, <standard deviation>), ... }
    """

    statistics = {}

    for id_measure in measures:
        acc = 0
        for measure in measures[id_measure]:
            acc += measure

        avg = acc / len(measures[id_measure])

        f_acc = 0
        for measure in measures[id_measure]:
            f_acc += (measure - avg) ** 2

        std_deviation = (f_acc / (len(measures[id_measure]) - 1)) ** 0.5

        statistics[id_measure] = (avg, std_deviation)

    return statistics
