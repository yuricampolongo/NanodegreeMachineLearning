#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sklearn
print 'The scikit-learn version is ', sklearn.__version__
if sklearn.__version__ >= '0.18':
    print "Você precisa fazer downgrade do scikit-learn ou ficar atento as diferenças nas versões citadas."
    print "Pode ser feito executando:\n"
    print "pip install scikit-learn==0.17"
else:
    print "Tudo certo!"