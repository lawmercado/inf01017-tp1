#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
import math


class DecisionTree(object):

    __dt = None

    def __init__(self, data_handler, attributes):
        self.__dt = self.__generate(data_handler, attributes)

    def __generate(self, data_handler, attributes):
        node = {"attr": None, "value": {}}

        by_class = data_handler.by_class_attribute_values()
        classes = list(by_class.keys())

        if len(classes) == 1:
            node["value"] = classes[0]

            return node

        if len(attributes) == 0:
            node["value"] = self.__get_most_occurred_class(data_handler)

            return node

        else:
            most_informative_attr = self.__get_most_informative_attr(data_handler, attributes)

            print("\nAtributo escolhido: " + most_informative_attr)

            node["attr"] = most_informative_attr

            attributes.remove(most_informative_attr)

            by_attributes = data_handler.by_attributes()

            values = list(set(by_attributes[most_informative_attr]))

            for value in values:
                print("\n" + most_informative_attr + " - valor: " + value)

                sub_data_handler = data_handler.filter_by_attr_value(most_informative_attr, value)

                node["value"][value] = self.__generate(sub_data_handler, attributes)

            return node

    def __information_gain(self, data_handler, attr):
        by_attributes = data_handler.by_attributes()

        value_count = {}

        total_values = len(by_attributes[attr])

        for value in by_attributes[attr]:
            if value in list(value_count):
                value_count[value] += 1
            else:
                value_count[value] = 1

        info_attr = 0

        for value in value_count:
            info = self.__information(data_handler.filter_by_attr_value(attr, value))
            info_attr += ((value_count[value] / total_values) * info)

        print("\nEntropia média para o atributo '" + attr + "': " + str(info_attr))

        info = self.__information(data_handler)

        return info - info_attr

    def __information(self, data_handler):
        data_by_class = data_handler.by_class_attribute_values()

        total_instances = len(data_handler.as_instances())

        info = 0

        for yi in data_by_class:
            pi = len(data_by_class[yi]) / total_instances

            info -= pi * math.log(pi, 2)

        return info

    def __get_most_informative_attr(self, data_handler, attributes):
        info_gain_by_attribute = {}

        for attr in attributes:
            info_gain = self.__information_gain(data_handler, attr)
            info_gain_by_attribute[attr] = info_gain

            print("Ganho de informação para o atributo '" + attr + "': " + str(info_gain))

        return max(info_gain_by_attribute, key=info_gain_by_attribute.get)

    def __get_most_occurred_class(self, data_handler):
        by_class = data_handler.by_class_attribute_values()

        most_occurred_class_count = max(len(value) for value in by_class.values())
        most_occurred_class = [k for k, value in by_class.items() if len(value) == most_occurred_class_count]

        return most_occurred_class

    def __tree_as_string(self, node, level):

        if node["attr"] is None:
            return ("\t" * level) + "-Class: " + node["value"] + "\n"

        else:
            text = ("\t" * level) + "-Attr: " + node["attr"] + "\n"

            for item in node["value"]:
                text += ("\t" * level) + "-Value: " + item + "\n"
                text += self.__tree_as_string(node["value"][item], (level + 1))

            return text

    def __str__(self):
        return "\nÁrvore gerada:\n" + self.__tree_as_string(self.__dt, 0).strip()
