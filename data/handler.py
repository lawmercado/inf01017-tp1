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

    def __init__(self, raw_data, class_attr, normalize=False):
        """
        Constructor of the class

        :param list raw_data: A list of data
        :param list class_attr: An attribute that contains important conclusion/information about the record
        """

        self.__data = list(raw_data)
        self.__attr = self.__data.pop(0)
        self.__class_attr = class_attr

        data = {}

        for idx_attr, attr in enumerate(self.__attr):
            data[attr] = []

            for row in self.__data:
                data[attr].append(self.__process_raw_data_value(row[idx_attr]))

        # Saves for further use
        if normalize:
            self.__data_by_attributes = self.__normalize(data)
        else:
            self.__data_by_attributes = data

    def attributes(self):
        return list(set(self.__attr).difference([self.__class_attr]))

    def by_attributes(self):
        if bool(self.__data_by_attributes):
            return dict(self.__data_by_attributes)

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

        averages = self.__get_averages(data)
        std_deviations = self.__get_std_deviations(data, averages)

        normalized_data = {key: [] for key in data}

        for key in data:
            for item in data[key]:
                try:
                    normalized_item = (item - averages[key]) / std_deviations[key]
                    normalized_data[key].append(normalized_item)
                except (TypeError, ZeroDivisionError):
                    normalized_data[key].append(item)

        return normalized_data

    def __get_averages(self, data):
        averages = {key: 0 for key in data}

        for key in data:
            try:
                for item in data[key]:
                    averages[key] += item

                averages[key] = averages[key] / len(data[key])
            except TypeError:
                pass

        return averages

    def __get_std_deviations(self, data, averages):
        std_deviations = {key: 0 for key in data}

        for key in data:
            try:
                for item in data[key]:
                    std_deviations[key] += (item - averages[key]) ** 2

                std_deviations[key] = (std_deviations[key] / (len(data[key]) - 1)) ** 0.5
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
            return dict(self.__data_as_instances)

        data = self.by_attributes()

        classes = data[self.__class_attr]
        data.pop(self.__class_attr)

        instances = []

        for x in range(0, len(classes)):
            instances.append(())

        for key in data:
            for idx_value, value in enumerate(data[key]):
                instances[idx_value] = instances[idx_value] + (value,)

        instances = [(instance, classes[idx_instance]) for idx_instance, instance in enumerate(instances)]

        # Saves for further use
        self.__data_as_instances = instances

        return list(self.__data_as_instances)

    def by_class_attr_values(self):
        instances = self.as_instances()

        data = {instance[1]: [] for instance in instances}

        for idx, instance in enumerate(instances):
            data[instance[1]].append(idx)

        return data

    def get_average_for_attr(self, attr):
        data = self.by_attributes()

        average = 0

        for item in data[attr]:
            average += item

        return average / len(data[attr])

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

    def filter_by_attr_value(self, attr, value, exp="=="):
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

        for idx_item, item in enumerate(by_attributes[attr]):
            expression = "item" + exp + "value"

            if not bool(eval(expression)):
                # +1 to avoid conflict when dealing with the data and it's attributes names
                to_remove_items.append(idx_item + 1)

        filtered_raw_data = [item for idx_item, item in enumerate(raw_data) if idx_item not in to_remove_items]

        return DataHandler(filtered_raw_data, self.__class_attr)

    def discretize(self):
        by_attributes = self.by_attributes()
        raw_data = self.as_raw_data()

        for attr in self.attributes():
            average = int(self.get_average_for_attr(attr))

            idx_attr = raw_data[0].index(attr)

            for idx_value in range(1, len(by_attributes[attr])):
                if by_attributes[attr][idx_value] <= average:
                    new_value = "<=" + str(average)
                else:
                    new_value = ">" + str(average)


                ### AQUIE TEM ALGUM PROBLEMINHA
                raw_data[idx_value][idx_attr] = new_value

        return DataHandler(raw_data, self.__class_attr)

    def as_raw_data(self):
        attributes = list([self.__attr])

        return list(attributes + list(self.__data))
