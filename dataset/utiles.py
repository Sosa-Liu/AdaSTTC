import numpy as np
import time
import datetime

class MinMaxNormalization(object):
    '''
    MinMax Normalization --> [-1, 1]
    x = (x - min) / (max - min).
    x = x * 2 - 1
    '''

    def __init__(self):
        pass

    def fit(self, X):
        self._min = X.min()
        self._max = X.max()
        print("min:", self._min, "max:", self._max)

    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        X = X * 2. - 1.
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = (X + 1.) / 2.
        X = 1. * X * (self._max - self._min) + self._min
        return X
    

class MinMaxNormalization_01(object):
    '''
    MinMax Normalization --> [0, 1]
    x = (x - min) / (max - min).
    '''

    def __init__(self):
        pass

    def fit(self, X):
        self._min = X.min()
        self._max = X.max()
        print("min:", self._min, "max:", self._max)

    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = 1. * X * (self._max - self._min) + self._min
        return X


def remove_incomplete_days(data, timestamps, T=48):
    '''remove a certain day which does not have 48 timestamps
    '''
    days = []  # available days: some day only contain some seqs
    days_incomplete = []
    i = 0
    while i < len(timestamps):
        if int(timestamps[i][8:]) != 1:
            i += 1
        elif i+T-1 < len(timestamps) and int(timestamps[i+T-1][8:]) == T:
            days.append(timestamps[i][:8])
            i += T
        else:
            days_incomplete.append(timestamps[i][:8])
            i += 1
    days = set(days)
    idx = []
    for i, t in enumerate(timestamps):
        if t[:8] in days:
            idx.append(i)

    data = data[idx]
    timestamps = [timestamps[i] for i in idx]
    timestamps = np.asarray(timestamps)

    return data, timestamps, days_incomplete


def string2timestamp(strings, T=48):
    '''
    convert string 'YYMMDDNN' to datetime.datetime(year, month, day, hour, minute)
    '''
    timestamps = []
    hour_per_slot = 24.0 / T 
    min_per_slot = 60.0 * hour_per_slot
    slots_per_hour = T // 24
    for t in strings:
        year, month, day, slot = int(t[:4]), int(t[4:6]), int(t[6:8]), int(t[8:]) - 1
        timestamps.append(datetime.datetime(year, month, day, hour = int(
            slot * hour_per_slot), minute= (slot % slots_per_hour) * int(min_per_slot)))
    return timestamps


def timestamp2string(timestamps, T=48):
    '''
    convert datetime.datetime(year, month, day, hour, minute) to string 'YYMMDDNN' 
    '''
    slots_per_hour = T // 24
    slots_per_min = slots_per_hour // 60
    strings = []
    for ts in timestamps:
        string = "%s%02i" % (ts.strftime('%Y%m%d'), 
                        int(1 + ts.hour*slots_per_hour + ts.minute*slots_per_min))
        strings.append(string)
    return strings

def check_breakpoints(timestamps, T=48):
    """
    return the index where there exists a missing slot before it
    """
    if type(timestamps[0]) != datetime.datetime:
        timestamps = string2timestamp(timestamps)

    offset = datetime.timedelta(minutes=24*60/T)
    breakpoints = [0]
    for i in range(1, len(timestamps)):
        if timestamps[i - 1] + offset != timestamps[i]:
            breakpoints.append(i)
    breakpoints.append(len(timestamps))

    return breakpoints

