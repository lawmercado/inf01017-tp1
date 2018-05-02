#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
import random


class DataHandler(object):
    """
    A class for raw data manipulation into specific structures

    """

    __attr = []
    __data = []
    __class_attr = ""
    __data_by_attributes = {}
    __data_as_instances = []

    def __init__(self, raw_data, class_attr):
        """
        Constructor of the class

        :param list raw_data: A list of data
        :param list class_attr: An attribute that contains important conclusion/information about the record
        """

        self.__data = list(raw_data)
        self.__attr = self.__data.pop(0)
        self.__class_attr = class_attr

    def attributes(self):
        return list(set(self.__attr).difference([self.__class_attr]))

    def by_attributes(self):
        if bool(self.__data_by_attributes):
            return self.__data_by_attributes

        data = {}

        for idx_attr, attr in enumerate(self.__attr):
            data[attr] = []

            for row in self.__data:
                try:
                    data[attr].append(self.__process_raw_data_value(row[idx_attr]))
                except ValueError:
                    pass

        # Saves for further use
        self.__data_by_attributes = data

        return data

    def __process_raw_data_value(self, record):
        return record.strip()

    def as_instances(self):
        """
        Convert the data to the attribute-classification format, aka: [((x11,...,x1n), y0),...,((xm1,...,xmn), ym)]
        which xij are the attributes of the instance and yi is the classification, based on the class attribute

        :return: A list of tuples
        :rtype: list
        """

        if self.__data_as_instances:
            return self.__data_as_instances

        data = self.by_attributes()

        classes = data[self.__class_attr]
        data.pop(self.__class_attr)

        normalized_data = self.normalize()

        instances = []

        for x in range(0, len(classes)):
            instances.append(())

        for key in normalized_data:
            for idx_value, value in enumerate(normalized_data[key]):
                instances[idx_value] = instances[idx_value] + (value,)

        instances = [(instance, classes[idx_instance]) for idx_instance, instance in enumerate(instances)]

        # Saves for further use
        self.__data_as_instances = instances

        return instances

    def by_class_attribute_values(self):
        instances = self.as_instances()

        data = {instance[1]: [] for instance in instances}

        for idx, instance in enumerate(instances):
            data[instance[1]].append(idx)

        return data

    def normalize(self):
        return self.by_attributes()

    def stratify(self, k_folds):
        """
        Divide the data into k folds, maintaining the main proportion

        :param integer k_folds: Number of folds
        :return: The folds
        :rtype: list
        """

        random.seed(None)

        instances = self.as_instances()
        data = self.by_class_attribute_values()

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

    def filter_by_attr_value(self, attr, value):
        """
        Generates a new DataHandler, with the data filtered by the attribute value

        :param string attr: Attribute name do filter by
        :param mixed value: Value of the attribute
        :return: A DataHandler with the filtered data
        :rtype: DataHandler
        """

        raw_data = self.__as_raw_data()

        data_handler = DataHandler(raw_data, self.__class_attr)

        by_attributes = data_handler.by_attributes()

        to_remove_items = []

        for idx_item, item in enumerate(by_attributes[attr]):
            if item == value:
                # +1 to avoid conflict when dealing with the data and it's attributes names
                to_remove_items.append(idx_item + 1)

        filtered_raw_data = [item for idx_item, item in enumerate(raw_data) if idx_item not in to_remove_items]

        return DataHandler(filtered_raw_data, self.__class_attr)

    def __as_raw_data(self):
        attributes = list([self.__attr])

        return attributes + list(self.__data)


class NumericDataHandler(DataHandler):
    """
    A class for numeric raw data manipulation into specific structures

    """

    def __process_raw_data_value(self, record):
        return float(record.strip())

    def __get_average(self):
        data = self.by_attributes()

        averages = {key: 0 for key in data}

        for key in data:
            for item in data[key]:
                averages[key] += item

            averages[key] = averages[key] / len(data[key])

        return averages

    def __get_std_deviations(self, averages):
        data = self.by_attributes()

        std_deviations = {key: 0 for key in data}

        for key in data:
            for item in data[key]:
                std_deviations[key] += (item - averages[key]) ** 2

            std_deviations[key] = (std_deviations[key] / (len(data[key]) - 1)) ** 0.5

        return std_deviations

    def normalize(self):
        """
        Normalizes the data represented by an dictionary

        :return: A list with the normalized data
        :rtype: list
        """

        data = self.by_attributes()

        averages = self.__get_average()
        std_deviations = self.__get_std_deviations(averages)

        normalized_data = {key: [] for key in data}

        for key in data:
            for item in data[key]:
                normalized_item = (item - averages[key]) / std_deviations[key]
                normalized_data[key].append(normalized_item)

        return normalized_data