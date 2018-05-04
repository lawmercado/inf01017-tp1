#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
import logging
import random
import copy

logger = logging.getLogger(__name__)


class DataHandler(object):
    """
    A class for raw data manipulation into specific structures

    """

    __header = []
    __data = []
    __class_attr = ""
    __idx_class_attr = -1
    __data_by_attr = []
    __data_as_instances = []

    def __init__(self, raw_data, class_attr, normalize=False):
        """
        Constructor of the class

        :param list raw_data: A list of data
        :param list class_attr: An attribute that contains important conclusion/information about the record
        """

        self.__data = copy.deepcopy(raw_data)
        self.__header = self.__data.pop(0)
        self.__idx_class_attr = self.__header.index(class_attr)
        self.__class_attr = class_attr

        data_by_attr = ()

        for idx_attr, attr in enumerate(self.__header):
            row_by_attr = []

            for row in self.__data:
                row_by_attr.append(self.__process_raw_data_value(row[idx_attr]))

            data_by_attr = data_by_attr + (row_by_attr,)

        # Saves for further use
        if normalize:
            self.__data_by_attr = self.__normalize(data_by_attr)
        else:
            self.__data_by_attr = data_by_attr

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

    def get_average_for_attr(self, attr):
        data = self.by_attributes()

        average = 0

        for item in data[self.attributes().index(attr)]:
            average += item

        return average / len(data[self.attributes().index(attr)])

    def stratify(self, k_folds):
        """
        Divide the data into k folds, maintaining the main proportion

        :param integer k_folds: Number of folds
        :return: The folds
        :rtype: list
        """

        random.seed(None)

        instances = self.as_instances()
        data = self.by_class_attr_values()

        folds = [[] for i in range(0, k_folds)]

        instances_per_fold = round(len(instances) / k_folds)

        for yi in data:
            yi_proportion = len(data[yi]) / len(instances)

            counter = round(yi_proportion * instances_per_fold)

            while counter > 0:
                try:
                    for idx in range(0, k_folds):
                        instance = instances[data[yi].pop(random.randint(0, len(data[yi]) - 1))]

                        folds[idx].append(instance)

                    counter -= 1

                except (ValueError, IndexError):
                    break

        return folds

    def bootstrap(self, ratio=1.0):
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
                average = float("{0:.3f}".format(self.get_average_for_attr(attr)))

                idx_attr = raw_data[0].index(attr)

                for idx_value in range(1, len(raw_data)):
                    if by_attributes[self.attributes().index(attr)][idx_value - 1] <= average:
                        new_value = "<=" + str(average)
                    else:
                        new_value = ">" + str(average)

                    raw_data[idx_value][idx_attr] = new_value
            except TypeError as e:
                logger.error(e)

        return DataHandler(raw_data, self.__class_attr)

    def as_raw_data(self):
        attributes = copy.deepcopy([self.__header])
        data = copy.deepcopy(self.__data)

        return list(attributes + data)

    def __str__(self):
        return str(self.by_attributes())