import numpy as np
import sys
sys.path.append("E:\\AdaSTTP")


def triangular(x_list, a, b, c):
    """
    Triangular fuzzy membership function 

    Input:
    x_list (float): a iterable list or ndarray of input value
    a (float): left boundary
    b (float): center point
    c (float): right boundary

    Output:
    float: degree of membership in the fuzzy set
    """
    y = []
    for x in x_list:
        if x <= a or x >= c:
            yy = 0
        elif a < x <= b:
            yy = (x - a) / (b - a)
        elif b < x < c:
            yy = (c - x) / (c - b)
        y.append(yy)
    return np.asarray(y)


def asymmetric_gaussian(x_list, left_sigma, mean, right_sigma):
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
    y = []
    for x in x_list:
        if x < mean:
            yy = np.exp(-0.5*((x-mean)/left_sigma)**2)
        else:
            yy = np.exp(-0.5*((x-mean)/right_sigma)**2)
        y.append(yy)
    return np.asarray(y)

