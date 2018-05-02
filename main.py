#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
import csv
from data.handler import DataHandler
from ml.supervised.algorithms import dt

rows = list(csv.reader(open('dados_benchmark_ad.csv', 'r'), delimiter=';'))

data_handler = DataHandler(rows, 'Joga')

dt(data_handler)
