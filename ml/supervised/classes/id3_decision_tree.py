#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
import logging

logger = logging.getLogger("main")


class ID3DecisionTree(object):

    __dt = None

    def __init__(self, data_handler):
        logger.info("Generating tree...")

        self.__dt = self.__generate(data_handler, data_handler.attributes())

        logger.info("Generated tree: \n" + str(self))

    def __generate(self, data_handler, attributes):
        node = {"attr": None, "value": {}}

        by_class = data_handler.by_class_attr_values()
        classes = list(by_class.keys())

        if len(classes) == 1:
            node["value"] = classes[0]

            return node

        if len(attributes) == 0:
            node["value"] = data_handler.most_occurred_class()

            return node

        else:
            idx_most_informative_attr = self.__get_most_informative_attr(data_handler, attributes)
            most_informative_attr = data_handler.attributes()[idx_most_informative_attr]

            logger.debug("Chosen attr: " + most_informative_attr)

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
                    node["value"] = data_handler.most_occurred_class()

                    return node

                node["value"][value] = self.__generate(sub_data_handler, attributes)

            return node

    def __get_most_informative_attr(self, data_handler, attributes):
        info_gain_by_attribute = [0 for i in range(0, len(data_handler.attributes()))]

        average_gain = 0

        for attr in attributes:
            info_gain = data_handler.information_gain(attr)

            info_gain_by_attribute[data_handler.attributes().index(attr)] = info_gain

            average_gain += info_gain

            logger.debug("Info. gain for '" + attr + "': " + str(info_gain))

        return info_gain_by_attribute.index(max(info_gain_by_attribute))

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
                    expression = value.format(test_instance[node["attr"][0]])

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
