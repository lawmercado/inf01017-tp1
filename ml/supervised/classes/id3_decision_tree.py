#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
import logging
import math

logger = logging.getLogger("main")


class ID3DecisionTree(object):

    __dt = None

    def __init__(self, data_handler):
        logger.debug("Generating tree...")

        self.__dt = self.__generate(data_handler.discretize(), data_handler.attributes())

        logger.debug("Generated tree: \n" + str(self))

    def __generate(self, data_handler, attributes):
        node = {"attr": None, "value": {}}

        by_class = data_handler.by_class_attr_values()
        classes = list(by_class.keys())

        if len(classes) == 1:
            node["value"] = classes[0]

            return node

        if len(attributes) == 0:
            node["value"] = self.__get_most_occurred_class(data_handler)

            return node

        else:
            idx_most_informative_attr = self.__get_most_informative_attr(data_handler, attributes)
            most_informative_attr = data_handler.attributes()[idx_most_informative_attr]

            logger.debug("Choosen attr: " + most_informative_attr)

            node["attr"] = (idx_most_informative_attr, most_informative_attr)

            try:
                attributes.remove(most_informative_attr)
            except ValueError:
                # Quando o ganho é 0
                most_informative_attr = attributes[0]
                attributes = []

            by_attributes = data_handler.by_attributes()

            values = []

            for value in by_attributes[idx_most_informative_attr]:
                if value not in values:
                    values.append(value)

            for value in values:
                logger.debug("Analysing " + most_informative_attr + ": value: " + str(value))

                sub_data_handler = data_handler.filter_by_attr_value(most_informative_attr, value)

                if len(sub_data_handler.as_instances()) == 0:
                    node["attr"] = None
                    node["value"] = self.__get_most_occurred_class(data_handler)

                    return node

                node["value"][value] = self.__generate(sub_data_handler, attributes)

            return node

    def __get_most_informative_attr(self, data_handler, attributes):
        info_gain_by_attribute = [0 for i in range(0, len(data_handler.attributes()))]

        average_gain = 0

        for attr in attributes:
            info_gain = self.__information_gain(data_handler, attr)

            info_gain_by_attribute[data_handler.attributes().index(attr)] = info_gain

            average_gain += info_gain

            logger.debug("Info. gain for '" + attr + "': " + str(info_gain))

        return info_gain_by_attribute.index(max(info_gain_by_attribute))

    def __information_gain(self, data_handler, attr):
        by_attributes = data_handler.by_attributes()

        value_count = {}
        total_values = len(by_attributes[data_handler.attributes().index(attr)])
        info_attr = 0

        for value in by_attributes[data_handler.attributes().index(attr)]:
            if value in list(value_count):
                value_count[value] += 1
            else:
                value_count[value] = 1

        for value in value_count:
            info = self.__information(data_handler.filter_by_attr_value(attr, value))
            info_attr += ((value_count[value] / total_values) * info)

        logger.debug("Entropy for '" + attr + "': " + str(info_attr))

        info = self.__information(data_handler)

        return info - info_attr

    def __information(self, data_handler):
        data_by_class = data_handler.by_class_attr_values()

        total_instances = len(data_handler.as_instances())

        info = 0

        for yi in data_by_class:
            pi = len(data_by_class[yi]) / total_instances

            info -= pi * math.log(pi, 2)

        return info

    def __get_most_occurred_class(self, data_handler):
        by_class = data_handler.by_class_attr_values()

        most_occurred_class_count = max([len(value) for value in by_class.values()])
        most_occurred_class = [k for k, value in by_class.items() if len(value) == most_occurred_class_count]

        return most_occurred_class[0]

    def __tree_as_string(self, node, level):
        if node["attr"] is None:
            return ("|\t" * level) + "|Class: " + str(node["value"]) + "\n"

        else:
            text = ("|\t" * level) + "|Attr: " + str(node["attr"]) + "\n"

            for item in node["value"]:
                text += ("|\t" * (level + 1)) + "|Value: " + str(item) + "\n"
                text += self.__tree_as_string(node["value"][item], (level + 2))

            return text

    def classify(self, test_instance):
        node = self.__dt

        while node["attr"] is not None:
            bk_node = node

            for value in node["value"]:
                if isinstance(test_instance[node["attr"][0]], float):
                    expression = str(test_instance[node["attr"][0]]) + value

                    if bool(eval(expression)):
                        node = node["value"][value]
                        break

                if test_instance[node["attr"][0]] == value:
                    node = node["value"][value]
                    break

            # In case of no value match, force a change
            if node == bk_node:
                node = node["value"][value]

        return node["value"]

    def __str__(self):
        return self.__tree_as_string(self.__dt, 0).strip()
