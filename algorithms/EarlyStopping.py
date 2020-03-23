#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 09:56:43 2019

Adjusted from https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d.
"""

import numpy as np

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=50, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.num_good_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            print(' *- Training aborted: metric is nan.')
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            print(' *- Training aborted: exceeded patience.')
            return True

        return False
    
    def keep_best(self, metrics):
        if self.best is None:
            self.best = metrics
            return True

        if np.isnan(metrics):
            return AssertionError('Metric is nan')

        if self.is_better(metrics, self.best):
            self.best = metrics
            self.num_good_epochs = 0
            return True
        
        elif metrics == self.best:
            self.num_good_epochs += 1
            return not self.num_good_epochs >= self.patience
        
        elif metrics > self.best:
            self.num_good_epochs = 0
            return False

        return False
            

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)
