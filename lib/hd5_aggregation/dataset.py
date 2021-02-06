#!/bin/python3

import os
import fnmatch
import sqlite3
import hdf5_getters as hdf


class Dataset():
    def __init__(self):
        self.conn = sqlite3.connect('db/subset_track_metadata.db')
        self.c = _CONN.cursor()

    def get_training(self, exclude_feature=None):
        pass

    def get_testing(self, exclude_feature=None):
        pass
