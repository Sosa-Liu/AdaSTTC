import numpy as np

def get_triangular_params(time_position_list=[13, 18, 29, 34, 38, 45], T=48, offset=2):
    """
    Generate triangular membership functions for a given time position partition list.

    Input:
    - time_position_list: list of end timestamp of each period, maximum is T
    - T: the number of timestamps in one day
    - offset: length of overlapping between adjacent periods
    """
    a_list = []
    b_list = []
    c_list = []
    num_positions = len(time_position_list)

    for j in range(num_positions):                                                                                                                                                                                                                                                                                                                                             
        a = (time_position_list[(j - 1) % num_positions] - offset) % T
        c = (time_position_list[j] + offset) % T
        if time_position_list[(j - 1) % num_positions] > time_position_list[j]:
            b = 0.5 * (time_position_list[(j - 1) % num_positions] + time_position_list[j] + T) - T
        else:
            b = 0.5 * (time_position_list[(j - 1) % num_positions] + time_position_list[j])
        if b < a:
            a -= T
        elif c < b:
            c += T
        a_list.append(a)
        b_list.append(b)
        c_list.append(c)


    return a_list, b_list, c_list


def get_guassian_params(time_position_list=[13, 18, 29, 34, 38, 45], T=48, offset=0.4):
    """
    Generate guassian membership functions for a given time position partition list.

    Input:
    - time_position_list: list of end timestamp of each period, maximum is T
    - T: the number of timestamps in one day
    - offset: scale of overlapping between adjacent periods
    """
    a_list = []
    b_list = []
    c_list = []
    num_positions = len(time_position_list)

    for j in range(num_positions):
        if time_position_list[(j - 1) % num_positions] > time_position_list[j]:
            b = 0.5 * \
                (time_position_list[(j - 1) % num_positions] +
                    time_position_list[j] + T) - T
        else:
            b = 0.5 * \
                (time_position_list[(j - 1) % num_positions] + time_position_list[j])
        a = (b - time_position_list[(j - 2) % num_positions] + T) % T
        c = (time_position_list[(j + 1) % num_positions] - b + T) % T 
        a_list.append(offset * a)
        b_list.append(b)
        c_list.append(offset * c)

    return a_list, b_list, c_list


def triangular(x, a, b, c, T=48):
    """
    Triangular fuzzy membership function

    Input:
    x (float): input value
    a (float): left boundary
    b (float): center point
    c (float): right boundary

    Output:
    float: degree of membership in the fuzzy set
    """
    # link time slot T and next time slot 1
    if x > T - 1 + a:
        x -= T
    
    if x <= a or x >= c:
        return 0.
    elif a < x <= b:
        return (x - a) / (b - a)
    elif b < x < c:
        return (c - x) / (c - b)


def asymmetric_gaussian(x, left_sigma, mean, right_sigma, T=48):
    """
    Asymmetric Gaussian fuzzy membership function

    Input:
    x (float): input value
    mean (float): mean value of the fuzzy set
    left_sigma (float): standard deviation of the fuzzy set
    right_sigma (float): standard deviation of the fuzzy set

    Output:
    float: degree of membership in the fuzzy set
    """
    # link time slot T and next time slot 1
    if x > (T - 1 + mean - left_sigma):
        x -= T

    if x < mean:
        return np.exp(-0.5*((x-mean)/left_sigma)**2)
    else:
        return np.exp(-0.5*((x-mean)/right_sigma)**2)
