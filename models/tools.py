#!/usr/bin/python3
# -*- coding: utf-8 -*-
# =============================================================================
#         FILE: tools.py
#  DESCRIPTION:
#        USAGE:
#      OPTIONS:
# REQUIREMENTS:
#
#      LICENCE:
#
#         BUGS:
#        NOTES:
#       AUTHOR: Jiří Tumpach (tumpji),
# ORGANIZATION:
#      VERSION: 1.0
#      CREATED: 2019 08.02.
# =============================================================================

import tensorflow as tf
from abc import abstractmethod

class Logger:
    def __init__(self, verbose=0):
        self.verbose_level = verbose
    def log(self, level, msg):
        if self.verbose_level > level:
            print(msg)




class Classifier(Logger, tf.keras.Model):
    def __init__(self, **kwargs):
        Logger.__init__(self, **kwargs)
        tf.keras.Model.__init__(self)

    @abstractmethod
    def fit(X, Y):
        pass

    @abstractmethod
    def predict(X):
        pass
