import time
import numpy as np
import datetime

import sys
sys.path.append("E:\\AdaSTTP")
from fuzzy_utils import *
from dataset.utiles import string2timestamp, timestamp2string, check_breakpoints
from dataset.process_TaxiBJ import load_flow


def get_temporal_seq(datafile, closeness=4, p_daily=None, p_weekly=None, 
                     per_day = datetime.timedelta(days=1), per_week = datetime.timedelta(days=7)):
    
    data, timeslots = load_flow(datafile)
    timestamps = string2timestamp(timeslots)
    breakpoints = check_breakpoints(timestamps)
    start = timestamps[0] + p_weekly * per_week

    # timestamps index
    timestamp_idx = dict()
    for i, t in enumerate(timestamps):
        timestamp_idx[t] = i
        
    X = []
    Y = []
    target_timestamps = []

    for b in range(1, len(breakpoints)):
        idx = range(breakpoints[b-1], breakpoints[b])   # index of a continuous data
        # skip when its length is too short
        if (len(idx) < closeness) or (timestamps[idx[0]] < start):
            continue
        for i in range(len(idx) - closeness): 
            target_timestamp = timestamps[idx[i + closeness]]
            sample_index = []
            # sample weekly periodicity 
            if p_weekly != None:
                weekly_index = []
                for week in range(p_weekly, 0, -1):
                    weekly_timestamp = target_timestamp - week * per_week
                    if weekly_timestamp not in timestamp_idx:
                        break
                    weekly_index.append(timestamp_idx[weekly_timestamp])
                if len(weekly_index) != p_weekly:
                    continue
                sample_index += weekly_index

            # sample daily periodicity 
            if p_daily != None:
                daily_index = []
                for day in range(p_daily, 0, -1):
                    daily_timestamp = target_timestamp - day * per_day
                    if daily_timestamp not in timestamp_idx:
                        break
                    daily_index.append(timestamp_idx[daily_timestamp])
                if len(daily_index) != p_daily:
                    continue
                sample_index += daily_index
                
            sample_index += idx[i: i + closeness]

            x = np.vstack(np.expand_dims(data[sample_index], 0))
            y = data[idx[i + closeness]]
            X.append(x)
            Y.append(y)
            target_timestamps.append(target_timestamp)

    X = np.asarray(X)
    Y = np.asarray(Y)
    timeslots = timestamp2string(target_timestamps)
    return(X, Y, timeslots)


