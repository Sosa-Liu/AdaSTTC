import os
import numpy as np
import datetime

import sys
sys.path.append("E:\\AdaSTTP")
# from temporal_feature import *
from dataset.utiles import string2timestamp

class STDSFeature(object):
    """
    Spatial-Temporal Dynamic Shit Feature
    """
    def __init__(self, data, timeslots, T=48):
        super(STDSFeature, self).__init__()
        assert len(data) == len(timeslots)
        self.data = data
        self.timeslots = timeslots
        self.T = T
        self.timestamps = string2timestamp(timeslots, T=self.T)
        self.make_index()
        
    def make_index(self):
        self.get_index = dict()
        for i, ts in enumerate(self.timeslots):
            self.get_index[ts] = i
    
    def check_missing(timestamp):
        if timestamp not in self.get_index.keys():
            return True
        return False

    def check_complete(self):
        missing_timestamps = []
        offset = datetime.timedelta(minutes=24 * 60 // self.T)
        timestamps = self.timestamps
        i = 1
        while i < len(timestamps):
            if timestamps[i-1] + offset != timestamps[i]:
                missing_timestamps.append("(%s -- %s)" % (timestamps[i-1], timestamps[i]))
            i += 1
        for v in missing_timestamps:
            print(v)
        assert len(missing_timestamps) == 0

    def get_matrix(self, timeslot):
        index= self.get_index[timeslot]
        return self.data[index]
