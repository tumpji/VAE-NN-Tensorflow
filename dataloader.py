#!/usr/bin/python3
# -*- coding: utf-8 -*-
# =============================================================================
#         FILE: dataloader.py
#  DESCRIPTION: Create DatasetGenerator - iterator generates data based on weeks/size 
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
#      CREATED: 2019 03.31.
# =============================================================================
import os, re
import numpy as np

class DatasetGenerator:
    DATAFOLDER='./data/'

    def __init__(self, size_of_queue, 
            train_iterations=None, 
            train_data_size=None,
            queue_filter=None, # not implemented
            skip_weeks=0,
            train=False):

        assert size_of_queue > 0
        assert train_iterations > 0 or train_data_size > 0
        assert isinstance(train, bool)

        self.train = train
        self.size_of_queue = size_of_queue
        self.train_iterations = train_iterations
        self.train_data_size = train_data_size
        self.skip_weeks = skip_weeks

        self._discovery()


    def _queue_up(self, queue, newdata):
        if queue is None:
            return newdata
        else:
            full = np.concatenate([queue, newdata], axis=0)
            last = max(0, full.shape[0] - self.size_of_queue)
            if len(full.shape) == 2:
                return full[last:, :]
            elif len(full.shape) == 1:
                return full[last:]

    def queue_iterator(self, remove_partial_full=True, disable_queue_after_first_iteration=False):
        """ iterate over weeks and return actual week + it's history  (based on size of queue) """
        queue_features = None
        queue_labels = None
        training_phase = True
        Ndata, Niteration = 0, 0
        queue_disabled = False

        for awf, awl in self._week_iterator():
            do_yield = True

            if remove_partial_full and (queue_labels is None or queue_labels.shape[0] < self.size_of_queue):
                do_yield = False

            if do_yield:
                # some monitoring - when training stops ...
                Ndata += awl.shape[0]
                Niteration += 1

                if self.train == training_phase:
                    yield (queue_features, queue_labels), (awf, awl)
                    if disable_queue_after_first_iteration:
                        queue_disabled = True

            # switch phase
            if training_phase:
                if (self.train_iterations is not None and Niteration >= self.train_iterations) or \
                        (self.train_data_size is not None and Ndata >= self.train_data_size):
                    training_phase = False
                    queue_features = None
                    queue_labels = None
                    if self.train:
                        break

            if not queue_disabled:
                # add this week to queue
                queue_features = self._queue_up(queue_features, awf)
                queue_labels = self._queue_up(queue_labels, awl)

    def chunk_iterator(self):
        """ iterator for one big file """
        ### TT skip weeks
        file = np.load(self.chunk_file)
        avast = file['avast']
        features = file['features']

        skipped = 25000*self.skip_weeks
        for index in range(skipped, avast.shape[0]-self.size_of_queue+1, self.size_of_queue):
            yield features[index:index+self.size_of_queue, :], avast[index:index+self.size_of_queue]
                


    def _discovery(self):
        """ search for all weeks in database """
        self.weeks = []
        # search weeks in folder, extract integers
        reg = re.compile('^\d+')
        for name in os.listdir(self.DATAFOLDER):
            if not name.endswith('.npz'):
                continue
            if name == 'old.npz':
                self.chunk_file = self.DATAFOLDER + 'old.npz'
                continue
            if name == 'old_full.npz':
                self.chunk_file_full = self.DATAFOLDER + 'old_full.npz'
                continue
            integer = reg.match(name).group()
            if integer is None: continue
            else: integer = int(integer)
            self.weeks.append( (integer, self.DATAFOLDER + name) )
        # sort by week
        self.weeks = list(map(lambda x: x[1], # get path
            sorted(self.weeks))) # sort by weeknum
        self.weeks = self.weeks[self.skip_weeks:]

    def _week_iterator(self):
        """ iterate over weeks """
        for n, name_of_file in enumerate(self.weeks):
            actual_week = np.load(name_of_file)
            actual_week_features, actual_week_labels = actual_week['features'], actual_week['avast']
            yield actual_week_features, actual_week_labels
           
        

