#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
import csv
import logging
import sys

from data.handler import DataHandler
from ml.supervised.algorithms import dt
from ml.supervised.algorithms import random_trees

logging.basicConfig(format='%(levelname)s, %(name)s: %(message)s', stream=sys.stdout, level=logging.DEBUG)

logger = logging.getLogger(__name__)

rows = list(csv.reader(open('diabetes.csv', 'r'), delimiter=','))
data_handler = DataHandler(rows, 'Outcome')

'''rows = list(csv.reader(open('benchmark_dt.csv', 'r'), delimiter=';'))
data_handler = DataHandler(rows, 'Joga')'''

print(data_handler.header())
print(data_handler.attributes())
print(data_handler.by_attributes())
print(data_handler.as_instances())
print(data_handler.by_class_attr_values())


random_trees(data_handler, [], 1)
