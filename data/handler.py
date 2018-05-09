#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
import logging
import random
import math
import copy

logger = logging.getLogger("main")


class DataHandler(object):
    """
    A class for raw data manipulation into specific structures

    """

    __header = []
    __data = []
    __class_attr = None
    __idx_class_attr = None
    __data_by_attr = []
    __data_as_instances = []

    def __init__(self, raw_data, class_attr, id_attr=None, normalize=False):
        """
        Constructor of the class

        :param list raw_data: A list of data
        :param list class_attr: An attribute that contains important conclusion/information about the record
        """

        self.__data = copy.deepcopy(raw_data)
        self.__header = self.__data.pop(0)
        self.__class_attr = class_attr

        data_by_attr = []

        for idx_attr, attr in enumerate(self.__header):
            row_by_attr = []

            for row in self.__data:
                row_by_attr.append(self.__process_raw_data_value(row[idx_attr]))

            data_by_attr.append(row_by_attr)

        # If present, removes id column
        if id_attr is not None:
            idx_id_attr = self.__header.index(id_attr)

            data_by_attr.pop(idx_id_attr)
            self.__header.pop(idx_id_attr)

            for idx_data in range(0, len(self.__data)):
                self.__data[idx_data].pop(idx_id_attr)

        # After id column removal (if any), update the idx of class attr
        self.__idx_class_attr = self.__header.index(class_attr)

        # Moves the class column to the end of the data
        if self.__idx_class_attr != (len(self.__header) - 1):
            classes = data_by_attr.pop(self.__idx_class_attr)
            header_class_item = self.__header.pop(self.__idx_class_attr)

            for idx_data in range(0, len(self.__data)):
                item = self.__data[idx_data].pop(self.__idx_class_attr)
                self.__data[idx_data].append(item)

            data_by_attr.append(classes)
            self.__header.append(header_class_item)

            self.__idx_class_attr = self.__header.index(self.__class_attr)

        data_by_attr = tuple(data_by_attr)

        # Saves for further use
        if normalize:
            self.__data_by_attr = self.__normalize(data_by_attr)
        else:
            self.__data_by_attr = data_by_attr

    def __process_raw_data_value(self, record):
        value = record.strip()

        try:
            return float(value)

        except ValueError:
            return value

    def __normalize(self, data):
        """
        Normalizes the data represented by an dictionary

        :return: A list with the normalized data
        :rtype: list
        """

        header = self.header()
        averages = self.__get_averages(data)
        std_deviations = self.__get_std_deviations(data, averages)

        normalized_data = ()

        for idx_attr in range(0, len(data)):
            normalized_data = normalized_data + ([],)

        for idx_attr, attr_values in enumerate(data):
            for value in attr_values:
                try:
                    normalized_item = (value - averages[header[idx_attr]]) / std_deviations[header[idx_attr]]
                    normalized_data[idx_attr].append(float("{0:.3f}".format(normalized_item)))

                except (TypeError, ZeroDivisionError):
                    normalized_data[idx_attr].append(normalized_item)

        return normalized_data

    def __get_averages(self, data):
        header = self.header()
        averages = {header[i]: 0 for i in range(0, len(data))}

        for idx_attr, attr_values in enumerate(data):
            try:
                for value in attr_values:
                    averages[header[idx_attr]] += value

                averages[header[idx_attr]] = averages[header[idx_attr]] / len(attr_values)
            except TypeError:
                pass

        return averages

    def __get_std_deviations(self, data, averages):
        header = self.header()
        std_deviations = {header[i]: 0 for i in range(0, len(data))}

        for idx_attr, attr_values in enumerate(data):
            try:
                for value in attr_values:
                    std_deviations[header[idx_attr]] += (value - averages[header[idx_attr]]) ** 2

                std_deviations[header[idx_attr]] = (std_deviations[header[idx_attr]] / (len(attr_values) - 1)) ** 0.5
            except TypeError:
                pass

        return std_deviations

    def header(self):
        return list(self.__header)

    def attributes(self):
        attributes = list(self.__header)
        attributes.remove(self.__class_attr)

        return attributes

    def class_attribute(self):
        return self.__class_attr

    def by_attributes(self):
        if bool(self.__data_by_attr):
            return copy.deepcopy(self.__data_by_attr)

    def as_instances(self):
        """
        Convert the data to the attribute-classification format, aka: [((x11,...,x1n), y0),...,((xm1,...,xmn), ym)]
        which xij are the attributes of the instance and yi is the classification, based on the class attribute

        :return: A list of tuples
        :rtype: list
        """

        if self.__data_as_instances:
            return copy.deepcopy(self.__data_as_instances)

        data = self.by_attributes()

        classes = data[self.__idx_class_attr]

        instances = []

        for x in range(0, len(classes)):
            instances.append(())

        for idx_attr in range(0, len(self.attributes())):
            for idx_value, value in enumerate(data[idx_attr]):
                instances[idx_value] = instances[idx_value] + (value,)

        instances = [(instance, classes[idx_instance]) for idx_instance, instance in enumerate(instances)]

        # Saves for further use
        self.__data_as_instances = instances

        return copy.deepcopy(self.__data_as_instances)

    def by_class_attr_values(self):
        instances = self.as_instances()

        data = {instance[1]: [] for instance in instances}

        for idx, instance in enumerate(instances):
            data[instance[1]].append(idx)

        return data

    def as_raw_data(self):
        attributes = copy.deepcopy([self.__header])
        data = copy.deepcopy(self.__data)

        return list(attributes + data)

    def get_average_for_attr(self, attr):
        data = self.by_attributes()

        average = 0

        for item in data[self.attributes().index(attr)]:
            average += item

        return average / len(data[self.attributes().index(attr)])

    def possible_classes(self):
        instances = self.as_instances()

        classes = []
        seen = set()
        for instance in instances:
            if instance[1] not in seen:
                classes.append(instance[1])
                seen.add(instance[1])
        return classes

    def in_folds(self, k):
        """
        Divide the data into k folds
        :param k: Number of folds
        :return: The folds
        :rtype: List
        """
        random.seed(None)

        data = self.as_raw_data()
        classes = self.by_class_attr_values()

        # Remove the header
        data.pop(0)

        folds = [[] for i in range(k)]

        for i in range(len(data)):
            sample_index = random.randint(0, len(data) - 1)
            sample = data.pop(sample_index)
            folds[i % k].append(sample)

        return folds

    def fold_handler(self, folds):
        """
        Transform a list of folds into a DataHandler

        :param folds: A list of folds
        :return: A DataHandler containing all samples on the list
        :rtype: DataHandler
        """
        samples = []
        # Joins the list of folds into a list of samples
        for fold in folds:
            samples += fold

        samples.insert(0, self.__header)
        handler = DataHandler(samples, self.__class_attr)

        return handler

    def folds_handler(self, folds):
        """
        Transform a list of folds into a list of DataHandlers

        :param folds: A list of folds
        :return: A list of DataHandlers
        """
        folds_handler = [[] for i in range(len(folds))]

        for i in range(len(folds)):
            folds[i].insert(0, self.__header)
            folds_handler[i] = DataHandler(folds[i], self.__class_attr)

        return folds_handler

    def stratify(self, k_folds):
        """
        Divide the data into k stratified folds, maintaining the main proportion

        :param integer k_folds: Number of folds
        :return: The folds
        :rtype: list
        """

        random.seed(None)

        data = self.as_raw_data()
        classes = self.by_class_attr_values()

        folds = [[] for i in range(0, k_folds)]

        instances_per_fold = round(len(data) / k_folds)

        for yi in classes:
            yi_proportion = len(classes[yi]) / len(data)

            counter = round(yi_proportion * instances_per_fold)

            while counter > 0:
                try:
                    for idx in range(0, k_folds):
                        instance = data[classes[yi].pop(random.randint(0, len(classes[yi]) - 1))]

                        folds[idx].append(instance)

                    counter -= 1

                except (ValueError, IndexError):
                    break

        return folds

    def bootstrap(self, ratio=1.0):
        """
        Generates a list with refilling samples from the DataHandler's data

        :param ratio: Percentage of the data size tha defines the generated bootstrap size
        :return: A list containing the randomly chosen samples
        :rtype: List
        """
        data = self.as_raw_data()

        # Remove the header
        data.pop(0)

        bootstrap = []
        bootstrap_size = round((len(data) - 1) * ratio)

        while len(bootstrap) < bootstrap_size:
            index = random.randrange(len(data) - 1)
            bootstrap.append(data[index])

        return bootstrap

    def bagging(self, k):
        """
        Generates a list of bootstrap DataHandlers

        :param k: Number of bootstraps to be generated
        :return: A list containing k DataHandlers from k bootstraps
        :rtype: List of DataHandler
        """
        bootstraps = []

        for i in range(k):
            raw_bootstrap = list(list([self.__header]) + self.bootstrap())

            handler = DataHandler(raw_bootstrap, self.__class_attr)

            bootstraps.append(handler)

        return bootstraps

    def filter_by_attr_value(self, attr, value):
        """
        Generates a new DataHandler, with the data filtered by the attribute value

        :param string attr: Attribute name do filter by
        :param mixed value: Value of the attribute
        :return: A DataHandler with the filtered data
        :rtype: DataHandler
        """

        raw_data = self.as_raw_data()

        data_handler = DataHandler(raw_data, self.__class_attr)

        by_attributes = data_handler.by_attributes()

        to_remove_items = []

        for idx_value, attr_value in enumerate(by_attributes[self.attributes().index(attr)]):
            if attr_value != value:
                # +1 to avoid conflict when dealing with the data and it's attributes names
                to_remove_items.append(idx_value + 1)

        filtered_raw_data = [item for idx_item, item in enumerate(raw_data) if idx_item not in to_remove_items]

        return DataHandler(filtered_raw_data, self.__class_attr)

    def discretize(self):
        by_attributes = self.by_attributes()
        raw_data = self.as_raw_data()

        for attr in self.attributes():
            try:
                idx_attr = raw_data[0].index(attr)

                average = float("{0:.3f}".format(self.get_average_for_attr(attr)))

                for idx_value in range(1, len(raw_data)):
                    if by_attributes[self.attributes().index(attr)][idx_value - 1] <= average:
                        new_value = "%f<=" + str(average)
                    else:
                        new_value = "%f>" + str(average)

                    raw_data[idx_value][idx_attr] = new_value

            except TypeError:
                pass

        return DataHandler(raw_data, self.__class_attr)

    def discretize_information_gain(self):
        by_attributes = self.by_attributes()
        raw_data = self.as_raw_data()
        test_raw_data = self.as_raw_data()

        for attr in self.attributes():
            try:
                idx_attr = raw_data[0].index(attr)

                values = list(by_attributes[idx_attr])

                values.sort()

                for i in range(0, len(values) - 1):
                    values[i] = (values[i] + values[i + 1] / 2)

                values = list(set(values))

                values.sort()

                value_most_gain = 0
                most_gain = 0

                for compare in values:
                    compare = float("{0:.3f}".format(compare))

                    idx_attr = raw_data[0].index(attr)

                    for idx_value in range(1, len(raw_data)):
                        if by_attributes[self.attributes().index(attr)][idx_value - 1] <= compare:
                            new_value = "%f<=" + str(compare)
                        else:
                            new_value = "%f>" + str(compare)

                        test_raw_data[idx_value][idx_attr] = new_value

                    gain = DataHandler(test_raw_data, self.__class_attr).information_gain(attr)

                    if gain > most_gain:
                        value_most_gain = compare
                        most_gain = gain

                for idx_value in range(1, len(raw_data)):
                    if by_attributes[self.attributes().index(attr)][idx_value - 1] <= value_most_gain:
                        new_value = "%f<=" + str(value_most_gain)
                    else:
                        new_value = "%f>" + str(value_most_gain)

                    raw_data[idx_value][idx_attr] = new_value

            except TypeError:
                pass

        return DataHandler(raw_data, self.__class_attr)

    def discretize_quartile(self):
        by_attributes = self.by_attributes()
        raw_data = self.as_raw_data()

        for attr in self.attributes():
            try:
                idx_attr = raw_data[0].index(attr)

                values = list(by_attributes[idx_attr])
                values.sort()

                quartiles = self.generate_quartiles(values)

                for idx_value in range(1, len(raw_data)):
                    value = by_attributes[self.attributes().index(attr)][idx_value - 1]

                    if value <= quartiles[0]:
                        new_value = "%f<=" + str(quartiles[0])
                    elif quartiles[0] < value <= quartiles[1]:
                        new_value = str(quartiles[0]) + "<%f<=" + str(quartiles[1])
                    elif quartiles[1] < value <= quartiles[2]:
                        new_value = str(quartiles[1]) + "<%f<=" + str(quartiles[2])
                    else:
                        new_value = "%f>" + str(quartiles[2])

                    raw_data[idx_value][idx_attr] = new_value

            except TypeError:
                pass

        return DataHandler(raw_data, self.__class_attr)

    def generate_quartiles(self, values):
        n = len(values)
        q1 = self.get_median(values[0:math.floor(n / 2)])
        q2 = self.get_median(values)
        q3 = self.get_median(values[math.ceil(n / 2): n])
        quartiles = [q1, q2, q3]

        return quartiles

    def get_median(self, reference):
        n = len(reference)

        if n % 2 == 0:
            q = (reference[int(n / 2)] + reference[int(n / 2) - 1]) / 2
        else:
            q = reference[int(n / 2)]

        return q

    def information_gain(self, attr):
        by_attributes = self.by_attributes()

        value_count = {}
        total_values = len(by_attributes[self.attributes().index(attr)])
        info_attr = 0

        for value in by_attributes[self.attributes().index(attr)]:
            if value in list(value_count):
                value_count[value] += 1
            else:
                value_count[value] = 1

        for value in value_count:
            info = self.filter_by_attr_value(attr, value).entropy()
            info_attr += ((value_count[value] / total_values) * info)

        logger.debug("Mean entropy for '" + attr + "': " + str(info_attr))

        info = self.entropy()

        return info - info_attr

    def entropy(self):
        data_by_class = self.by_class_attr_values()

        total_instances = len(self.as_instances())

        info = 0

        for yi in data_by_class:
            pi = len(data_by_class[yi]) / total_instances

            info -= pi * math.log(pi, 2)

        return info

    def most_occurred_class(self):
        by_class = self.by_class_attr_values()

        most_occurred_class_count = max([len(value) for value in by_class.values()])
        most_occurred_class = [k for k, value in by_class.items() if len(value) == most_occurred_class_count]

        try:
            return most_occurred_class[random.randint(0, 1)]
        except IndexError:
            return most_occurred_class[0]

    def __str__(self):
        return str(self.by_attributes())
