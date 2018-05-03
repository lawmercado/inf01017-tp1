#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
import csv
from data.handler import DataHandler
from ml.supervised.algorithms import dt

rows = list(csv.reader(open('dados_benchmark_ad.csv', 'r'), delimiter=';'))

data_handler = DataHandler(rows, 'Joga')

bootstrap = data_handler.make_bootstrap()
print("\nOne bootstrap list: \n")
print(bootstrap)
bootstrap_handler = data_handler.bootstrap_handler(bootstrap)
print("\nThe bootstrap above as DataHandler: \n")
print(bootstrap_handler)
bag = data_handler.bagging(5)
print("\nA list of 5 other bootstrap handlers: \n")
print(bag)

#print("\nÁrvore gerada:")
#print(dt(data_handler))
