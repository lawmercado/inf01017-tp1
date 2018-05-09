#! /usr/bin/python

from __future__ import division
from __future__ import print_function
from ml.supervised.algorithms import knn_classification, id3_decision_tree, id3_random_forest


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


def decision_tree_kcrossvalidation(data_handler, k_folds):
    """
    :param data_handler: Raw data for the cross validation
    :param k_folds: Number of folds to generate
    :return: List of tuples with values for accuracy and the F-measure
    """
    folds = data_handler.stratify(k_folds)
    folds_measures = []

    for index_fold, fold in enumerate(folds):
        aux_folds = list(folds)  # Copy the folds
        test_fold = [aux_folds.pop(index_fold)]

        test_handler = data_handler.fold_handler(test_fold)
        train_handler = data_handler.fold_handler(aux_folds)

        # Train the algorithm & Classify the test fold
        test_instances = [instance[0] for instance in test_handler.as_instances()]
        classified_samples = id3_decision_tree(train_handler, test_instances)

        measures = validate(classified_samples, test_handler.as_instances(), train_handler.possible_classes())
        folds_measures.append(measures)
    return folds_measures


def random_forest_kcrossvalidation(data_handler, k_folds, k_trees):
    """
    :param data_handler: Raw data for the cross validation
    :param k_folds: Number of folds to generate
    :param k_trees: Number of trees in the forest
    :return: List of tuple with values for accuracy and the F-measure
    """
    folds = data_handler.stratify(k_folds)
    folds_measures = {"acc": [], "f-measure": [], "recall": [], "precision": []}

    for index_fold, fold in enumerate(folds):
        aux_folds = list(folds)  # Copy the folds
        test_fold = [aux_folds.pop(index_fold)]

        test_handler = data_handler.fold_handler(test_fold)
        train_handler = data_handler.fold_handler(aux_folds)

        # Train the algorithm & Classify the test fold
        test_instances = [instance[0] for instance in test_handler.as_instances()]
        classified_samples = id3_random_forest(train_handler, test_instances, k_trees)

        measures = validate(classified_samples, test_handler.as_instances(), train_handler.possible_classes())
        folds_measures["acc"].append(measures["acc"])
        folds_measures["f-measure"].append((measures["f-measure"]))
        folds_measures["recall"].append(measures["recall"])
        folds_measures["precision"].append(measures["precision"])
    return folds_measures


def validate(predicted_samples, test_samples, classes):

    measures = {}

    # Initialize confusion matrix
    correct_classifications = 0
    true_positives = {}
    true_negatives = {}
    false_positives = {}
    false_negatives = {}
    for a_class in classes:
        true_positives[a_class] = 0
        true_negatives[a_class] = 0
        false_positives[a_class] = 0
        false_negatives[a_class] = 0

    # Compare classified samples with the test set
    for (predicted_sample, test_sample) in zip(predicted_samples, test_samples):
        # If right prediction
        if predicted_sample[1] == test_sample[1]:
            correct_classifications += 1
            true_positives[predicted_sample[1]] += 1

            for a_class in classes:
                if a_class != predicted_sample[1]:
                    true_negatives[a_class] += 1
        # If wrong
        else:
            for a_class in classes:
                if a_class == predicted_sample[1]:
                    false_positives[a_class] += 1
                else:
                    false_negatives[a_class] += 1

    # Generate the statistics for each class
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    for a_class in classes:
        total_true_positives += true_positives[a_class]
        total_false_positives += false_positives[a_class]
        total_false_negatives += false_negatives[a_class]
        if len(classes) <= 2:
            break

    acc = correct_classifications / len(predicted_samples)
    measures["acc"] = acc

    # Micro average for 3 or more possible classes
    rev = total_true_positives / (total_true_positives + total_false_negatives)
    prec = total_true_positives / (total_true_positives + total_false_positives)
    measures["recall"] = rev
    measures["precision"] = prec

    if correct_classifications == 0:
        f_measure = 0
    else:
        f_measure = 2 * (prec * rev) / (prec + rev)
    measures["f-measure"] = f_measure

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
