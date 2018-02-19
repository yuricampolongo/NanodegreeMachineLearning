#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importar as bibliotecas necessárias para este projeto
import numpy as np
import pandas as pd
import visuals as vs  # Supplementary code
from sklearn.cross_validation import ShuffleSplit

# Formatação mais bonita para os notebooks
%matplotlib inline

# Executar o conjunto de dados de imóveis de Boston
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis=1)

# Êxito
print "O conjunto de dados de imóveis de Boston tem {} pontos com {} variáveis em cada.".format(*data.shape)