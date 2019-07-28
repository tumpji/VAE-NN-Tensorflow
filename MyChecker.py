#!/usr/bin/python3
# -*- coding: utf-8 -*-
# =============================================================================
#         FILE: MyChecker.py
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
#      CREATED: 2019 06.24.
# =============================================================================

import numpy as np

class NanChecker:
    class NanError(Exception): pass

    def check(self, data):
        return self.check_static(data)
    @staticmethod
    def check_static(data):
        if np.all(np.isfinite(data)):
            return data
        else:
            raise NanChecker.NanError()




class CheckManager:
    def __init__(self, posible_errors, types):
        self.posible_errors = posible_errors
        self.counter = 0
        self.done = False
        self.types = [types] if not isinstance(types, list) else types

    def first(self):
        return self.counter == 0
    def not_done(self):
        return not self.done

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if type in self.types:
            self.counter += 1
            if self.counter > self.posible_errors:
                return False
            print("Euncantered NaN trying again ({}-th time)".format(self.counter))
            return True
        self.done = True
        return False


if __name__ == '__main__':

    def f():
        pass


    c = CheckManager(3, NanChecker.NanError)
    while c.not_done():
        with c:
            print('1')
            #raise NanChecker.NanError()
        if c.error():
            print('Clear')




