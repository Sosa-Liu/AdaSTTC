import os
import h5py
import numpy as np

import sys
sys.path.append("E:\\AdaSTTP")
from fuzzy_utils import *

DATAPATH = 'f:/sosa_data/TaxiBJ/'


def load_flow(file_path):
    '''load inflow and outflow gride data from year 2013 to 2016

    Input:
    - file_path: Path of the flow data file, a string like '*/TaxiBJ/'

    Output:
    - data_flow: ndarray of all flow data with shape (22484, 2, 32, 32)
    - timeslot_flow: ndarray of all timeslots with shape (22484,)
    '''
    data_flow = []
    timeslot_flow = []
    for year in range(13, 17):
        file_name = os.path.join(file_path, f'BJ{str(year)}_M32x32_T30_InOut.h5')
        f_flow = h5py.File(file_name, 'r')
        data = f_flow['data'][()]
        timeslot = f_flow['date'][()]
        data_flow.append(data)
        timeslot_flow.append(timeslot)
        f_flow.close()
    data_flow = np.vstack(data_flow)
    timeslot_flow = np.concatenate(timeslot_flow,axis=0)

    return data_flow, timeslot_flow


def load_meteorology(timeslots, file_path):
    '''load file 'BJ_Meteorology.h5'

    Input:
    - timeslots: ndarray of timeslots with shape (#timeslots,) , (22484,) if load all data
    - file_path: Path of the meteorology data file, a string like '*/TaxiBJ/'

    Output:
    - meteorology_data:  ndarray with shape (#timeslots, 19), (22484, 19) if load all data
    '''
    f_meteorology = h5py.File(os.path.join(file_path, 'BJ_Meteorology.h5'), 'r')
    Timeslot = f_meteorology['date'][()]
    WindSpeed = f_meteorology['WindSpeed'][()]
    Weather = f_meteorology['Weather'][()]
    Temperature = f_meteorology['Temperature'][()]
    f_meteorology.close()

    M = dict()  # map timeslot to index
    for i, slot in enumerate(Timeslot):
        M[slot] = i

    WS = []  # WindSpeed
    WR = []  # Weather
    TE = []  # Temperature
    for slot in timeslots:
        predicted_id = M[slot]
        current_id = predicted_id - 1
        WS.append(WindSpeed[current_id])
        WR.append(Weather[current_id])
        TE.append(Temperature[current_id])

    WS = np.asarray(WS) # (?, 1)
    WR = np.asarray(WR) # (?, 17)
    TE = np.asarray(TE) # (?, 1)

    # 0-1 scale
    WS = 1. * (WS - WS.min()) / (WS.max() - WS.min())
    TE = 1. * (TE - TE.min()) / (TE.max() - TE.min())

    meteorology_data = np.hstack([WR, WS[:, None], TE[:, None]])    # concatenate all these attributes

    return meteorology_data


def load_holiday(timeslots, file_path):
    '''load file 'BJ_Holiday.txt' 
    
    Input:
    - timeslots: ndarray of timeslots with shape (22484,) if load all data
    - file_path: Path of the holiday data file, a string like '*/TaxiBJ/'

    Output:
    - is_holiday: ndarray of {0, 1} with shape (#timeslots, 1), 1 indicating whether current timeslot is holiday
    '''
    f_holiday = open(os.path.join(file_path, 'BJ_Holiday.txt'), 'r')
    holidays = f_holiday.read().split()
    holidays = set([h.strip() for h in holidays])   # remove the leading and trailing whitespace of the string
    is_holiday = np.zeros(len(timeslots))
    for i, slot in enumerate(timeslots):
        s = bytes.decode(slot[:8])  # Decode the bytes to string
        if s in holidays:
            is_holiday[i] = 1

    return is_holiday[:, None]


def time_of_day(timeslots, fuzzy=None, time_position_list=[13, 18, 29, 34, 38, 45], T=48):
    '''the time position in one day

    Input:
    - timeslots: A list or ndarray of timeslots with the format 'YYYYMMDDNN', 'NN' indicates the NN^th timeslot in the day 'YYYYMMDD'
    - fuzzy: None, 'triangular' or 'guassian', default None. 
    - time_position_list: None or a list of end time position of each period.

    Output:
    - time_meta: A list of float number indicating which time position period the current slot belongs to.
    '''

    time_of_day = [int(t[8:10]) for t in timeslots]    # integer [1, T]
    # number of positions in one day
    num_positions = len(time_position_list)
    time_position = []
    if fuzzy == None:
        for i in time_of_day:
            p = [0. for _ in range(num_positions)]
            for j in range(num_positions):
                if time_position_list[j] >= i:
                    p[j] = 1.
                    break
                elif time_position_list[-1] < i:
                    p[0] = 1.
            
            time_position.append(p)

    elif fuzzy == 'triangular':
        a_list, b_list, c_list = get_triangular_params(time_position_list)
        
    elif fuzzy == 'guassian':
        a_list, b_list, c_list = get_guassian_params(time_position_list)

    for i in time_of_day:
        p = [0. for _ in range(num_positions)]
        for j in range(num_positions):
            a, b, c = a_list[j], b_list[j], c_list[j]
            p[j] = triangular(i, a, b, c) if fuzzy == 'triangular' else asymmetric_gaussian(i, a, b, c)
        time_position.append(p)

    return np.asarray(time_position)


def day_of_week(timeslots):
    '''day of week, is weekday or not

    Input:
    - timestampes: A list or ndarray of timeslots with the format 'YYYYMMDDNN', 'NN' indicates the NN^th timeslot in the day 'YYYYMMDD'

    Output:
    - week_meta[:7]: one-hot encoding indicating the day of week
    - week_meta[7]: 0--weekend, 1--weekday
    '''
    day_of_week = [time.strptime(str(
        t[:8], encoding='utf-8'), '%Y%m%d').tm_wday for t in timeslots]  # tm_wday(): day of week, range [0, 6], Monday is 0
    week_position = []
    for i in day_of_week:
        v = [0 for _ in range(7)]
        v[i] = 1
        if i >= 5:
            v.append(0)  # weekend
        else:
            v.append(1)  # weekday
        week_position.append(v)

    return np.asarray(week_position)


def load_time_position(timeslots):
    '''time of day, day of week, and is weekday or not

    Input:
    - timeslots: A list or ndarray of timeslots with the format 'YYYYMMDDNN', 'NN' indicates the NN^th timeslot in the day 'YYYYMMDD'

    Output:
    - time_meta[:7]: one-hot encoding indicating the day of week
    - time_meta[7]: 0--weekend, 1--weekday
    - time_meta[8:]: membership degree of each time period 
    '''
    week_position = day_of_week(timeslots)
    time_position = time_of_day(timeslots)
    time_meta = week_position.concatenate(time_position, 0)
    
    return time_meta


def get_extra_feature(timeslots, use_meteorology=True, use_time_meta=True, use_holiday=True):
    '''
    load specified extra features according to timeslots
    '''
    extra_feature = []
    if use_meteorology:
        meteorology_feature = load_meteorology(timeslots, DATAPATH)
        extra_feature.append(meteorology_feature)
        # print('meteorol feature: ', meteorology_feature.shape) -- meteorol feature:  (#timeslots, 19)
    if use_holiday:
        holiday_feature = load_holiday(timeslots, DATAPATH)
        extra_feature.append(holiday_feature)
        # print('holiday feature:', holiday_feature.shape) -- holiday feature: (#timeslots, 1)
    if use_time_meta:
        time_position_feature = load_time_position(timeslots)
        extra_feature.append(time_position_feature)
        # print('time feature:', time_position_feature.shape)  -- time feature: (#timeslots, 8)


    extra_feature = np.hstack(extra_feature) if len(extra_feature) > 0 else np.asarray(extra_feature)
    # print('mete feature: ', meta_feature.shape) -- meta feature:  (#timeslots, 28)

    return extra_feature

