#!/usr/bin env python
# -*-coding:utf-8-*-
# title        :uf.py
# author       :Stone
# date         :20190702
#
# 考虑是否需要增加定义域判定
"""
This model is multi-objection test functions from utf1 to utf10

The calculation process mainly consists of two parts.Firstly,
convert the 1-d input (individual) into 2-d, simplify subsequ-
ent calculations, and count the number of bits in the decision
variable that participate in different calculation formulas.
Then, calculate the objective function value of each individual.

Input:
---------------------------------------------------------------
x : must be a list or array which is one-dimensional(individual)
    or two-dimensional(population)
 example :
 [1] : individual which decision variables is 1-d
 [1, 1] : individual which decision variables is 2-d
 [[1,1], [1,1]] : population which decision variables is 2-d
 ......
N : int
 hyperparameter
eps : float
 hyperparameter

Output:
---------------------------------------------------------------
obj_func : array
        the objective functions value of x
"""
import numpy as np


def preprocess_two_obj(x):
    """
    This function preprocesses decision variables
    and calculates the number of odd bits and even bits
    -------------------------------------------------
    parameter
    x       :    list or array
            the value of decision variables
    return
    x         :    array
            2d-array
    even_count:    int
            the number of even bits
    odd_ count:    int
            the number of odd bits
    --------------------------------------------------
    """
    x = np.array(x)
    dim = x.ndim

    if dim == 1:
        x = np.array(x).reshape(-1, len(x))

    odd_count = 0
    even_count = 0

    for j in range(1, x.shape[1], 2):
        even_count += 1
    for j in range(2, x.shape[1], 2):
        odd_count += 1
    return x, even_count, odd_count


def preprocess_three_obj(x):
    """
    This function preprocesses decision variables
    and calculates the number of three parts
    -------------------------------------------------
    parameter
    x : list or array
     the value of decision variables which dimension is greater than one
    return
    x : 2-d array
     Row representative number
     Columns represent decision variable dimensions
    part1_count : int
               the number of bits which is a multiplication of four
    part2_count : int
               the number of bits which is a multiplication of five
    part3_count : int
               the number of bits which is a multiplication of three
    --------------------------------------------------
    """
    x = np.array(x)
    dim = x.ndim

    if dim == 1:
        x = np.array(x).reshape(-1, len(x))

    part1_count = 0
    part2_count = 0
    part3_count = 0

    for j in range(3, x.shape[1], 3):
        part1_count += 1
    for j in range(4, x.shape[1], 3):
        part2_count += 1
    for j in range(2, x.shape[1], 3):
        part3_count += 1
    return x, part1_count, part2_count, part3_count


def cal_obj_func_utf1(x):
    """
    This function calculates objective functions value
    -------------------------------------------------
    parameter
    x : list or array
     the value of decision variables
    return
    obj_func : array
            the value of objective functions
    --------------------------------------------------
    """
    x, even_count, odd_count = preprocess_two_obj(x)

    m = x.shape[0]  # 种群规模（输入样本个数）
    n = x.shape[1]  # 决策变量维度

    obj_func = np.zeros((m, 3))

    for i in range(0, m):
        temp_even = 0
        temp_odd = 0
        for j in range(1, n, 2):
            temp_even = temp_even + (x[i][j] - np.sin(6 * np.pi * x[i][0] +
                                     (j + 1) * np.pi / n)) ** 2
        if even_count != 0:
            temp_even = 2 * temp_even / even_count
        else:
            temp_even = 0  # 避免偶数位集合为空，分母为零的情况
        for j in range(2, n, 2):
            temp_odd = temp_odd + (x[i][j] - np.sin(6 * np.pi * x[i][0] +
                                   (j + 1) * np.pi / n)) ** 2
        if odd_count != 0:
            temp_odd = 2 * temp_odd / odd_count
        else:
            temp_odd = 0  # # 避免奇数位集合为空，分母为零的情况
        obj_func[i][0] = x[i][0] + temp_odd  # 目标函数1的值
        obj_func[i][1] = 1 - np.sqrt(x[i][0]) + temp_even  # 目标函数2的值
        obj_func[i][2] = obj_func[i][0] + obj_func[i][1]
    return obj_func


def cal_obj_func_utf2(x):
    """
    This function calculates objective functions value
    -------------------------------------------------
    parameter
    x : list or array
     the value of decision variables
    return
    obj_func : array
            the value of objective functions
    --------------------------------------------------
    """
    x, even_count, odd_count = preprocess_two_obj(x)

    m = x.shape[0]
    n = x.shape[1]

    obj_func = np.zeros((m, 3))

    for i in range(0, m):
        temp_even = 0
        temp_odd = 0
        for j in range(1, n, 2):
            temp = 0
            temp = 0.3 * (x[i][0] ** 2) * np.cos(24 * np.pi * x[i][0] + 4 *
                                                 (j + 1) * np.pi / n)
            temp = temp + 0.6 * x[i][0]
            temp = temp * np.sin(6 * np.pi * x[i][0] + (j + 1) * np.pi / n)
            temp_even = temp_even + (x[i][j] - temp) ** 2
        if even_count != 0:
            temp_even = 2 * temp_even / even_count
        else:
            temp_even = 0
        for j in range(2, n, 2):
            temp = 0
            temp = 0.3 * (x[i][0] ** 2) * np.cos(24 * np.pi * x[i][0] + 4 *
                                                 (j + 1) * np.pi / n)
            temp = temp + 0.6 * x[i][0]
            temp = temp * np.cos(6 * np.pi * x[i][0] + (j + 1) * np.pi / n)
            temp_odd = temp_odd + (x[i][j] - temp) ** 2
        if odd_count != 0:
            temp_odd = 2 * temp_odd / odd_count
        else:
            temp_odd = 0
        obj_func[i][0] = x[i][0] + temp_odd
        obj_func[i][1] = 1 - np.sqrt(x[i][0]) + temp_even
        obj_func[i][2] = obj_func[i][0] + obj_func[i][1]
    return obj_func


def cal_obj_func_utf3(x):
    """
    This function calculates objective functions value
    -------------------------------------------------
    parameter
    x : list or array
     the value of decision variables
    return
    obj_func : array
            the value of objective functions
    --------------------------------------------------
    """
    x, even_count, odd_count = preprocess_two_obj(x)

    m = x.shape[0]
    n = x.shape[1]

    obj_func = np.zeros((m, 3))

    for i in range(0, m):
        y = np.zeros((1, n))
        for j in range(1, n):
            exponent = 0.5 * (1 + 3 * (j - 1) / (n - 2))
            y[0][j] = np.power(x[i][0], exponent)
            y[0][j] = x[i][j] - y[0][j]
        temp_even_sum = 0
        temp_even_mul = 1
        temp_odd_sum = 0
        temp_odd_mul = 1
        for j in range(1, n, 2):
            temp_even_sum = temp_even_sum + y[0][j] ** 2
            temp_even_mul = temp_even_mul * np.cos(20 * y[0][j] * np.pi /
                                                   (np.sqrt(j + 1)))
        if even_count != 0:
            temp_even = 2 * (4 * temp_even_sum - 2 *
                             temp_even_mul + 2) / even_count
        else:
            temp_even = 0
        for j in range(2, n, 2):
            temp_odd_sum = temp_odd_sum + y[0][j] ** 2
            temp_odd_mul = temp_odd_mul * np.cos(20 * y[0][j] * np.pi /
                                                 (np.sqrt(j + 1)))
        if odd_count != 0:
            temp_odd = 2 * (4 * temp_odd_sum - 2 *
                            temp_odd_mul + 2) / odd_count
        else:
            temp_odd = 0
        obj_func[i][0] = x[i][0] + temp_odd
        obj_func[i][1] = 1 - np.sqrt(x[i][0]) + temp_even
        obj_func[i][2] = obj_func[i][0] + obj_func[i][1]
    return obj_func


def cal_obj_func_utf4(x):
    """
    This function calculates objective functions value
    -------------------------------------------------
    parameter
    x : list or array
     the value of decision variables
    return
    obj_func : array
            the value of objective functions
    --------------------------------------------------
    """
    x, even_count, odd_count = preprocess_two_obj(x)

    m = x.shape[0]
    n = x.shape[1]

    obj_func = np.zeros((m, 3))

    for i in range(0, m):
        y = np.zeros((1, n))
        for j in range(1, n):
            y[0][j] = x[i][j] - np.sin(6 * np.pi * x[i][0] +
                                       (j + 1) * np.pi / n)
            y[0][j] = np.abs(y[0][j]) / (1 + np.exp(2 * np.abs(y[0][j])))
        temp_even = 0
        temp_odd = 0
        for j in range(1, n, 2):
            temp_even = temp_even + y[0][j]
        if even_count != 0:
            temp_even = 2 * temp_even / even_count
        else:
            temp_even = 0
        for j in range(2, n, 2):
            temp_odd = temp_odd + y[0][j]
        if odd_count != 0:
            temp_odd = 2 * temp_odd / odd_count
        else:
            temp_odd = 0
        obj_func[i][0] = x[i][0] + temp_odd
        obj_func[i][1] = 1 - (x[i][0] ** 2) + temp_even
        obj_func[i][2] = obj_func[i][0] + obj_func[i][1]
    return obj_func


def cal_obj_func_utf5(x, N=10, eps=0.1):
    """
    This function calculates objective functions value
    -------------------------------------------------
    parameter
    x : list or array
     the value of decision variables
    N : int
     hyperparameter
    eps : float
       hyperparameter
    return
    obj_func : array
            the value of objective functions
    --------------------------------------------------
    """
    x, even_count, odd_count = preprocess_two_obj(x)

    m = x.shape[0]
    n = x.shape[1]

    obj_func = np.zeros((m, 3))

    for i in range(0, m):
        y = np.zeros((1, n))
        for j in range(1, n):
            y[0][j] = x[i][j] - np.sin(6 * np.pi * x[i][0] +
                                       (j + 1) * np.pi / n)
            y[0][j] = 2 * (y[0][j] ** 2) - np.cos(4 * np.pi * y[0][j]) + 1
        temp_even = 0
        temp_odd = 0
        for j in range(1, n, 2):
            temp_even = temp_even + y[0][j]
        if even_count != 0:
            temp_even = 2 * temp_even / even_count
        else:
            temp_even = 0
        for j in range(2, n, 2):
            temp_odd = temp_odd + y[0][j]
        if odd_count != 0:
            temp_odd = 2 * temp_odd / odd_count
        else:
            temp_odd = 0
        obj_func[i][0] = (x[i][0] + (0.5 / N + eps) * np.
                          abs(np.sin(2 * N * np.pi * x[i][0])) + temp_odd)
        obj_func[i][1] = (1 - x[i][0] + (0.5 / N + eps) * np.
                          abs(np.sin(2 * N * np.pi * x[i][0])) + temp_even)
        obj_func[i][2] = obj_func[i][0] + obj_func[i][1]
    return obj_func


def cal_obj_func_utf6(x, N=2, eps=0.1):
    """
    This function calculates objective functions value
    -------------------------------------------------
    parameter
    x : list or array
     the value of decision variables
    N : int
     hyperparameter
    eps : float
       hyperparameter
    return
    obj_func : array
            the value of objective functions
    --------------------------------------------------
    """
    x, even_count, odd_count = preprocess_two_obj(x)

    m = x.shape[0]
    n = x.shape[1]

    obj_func = np.zeros((m, 3))

    for i in range(0, m):
        y = np.zeros((1, n))
        for j in range(1, n):
            y[0][j] = x[i][j] - np.sin(6 * np.pi * x[i][0] +
                                       (j + 1) * np.pi / n)
        temp_even_sum = 0
        temp_even_mul = 1
        temp_odd_sum = 0
        temp_odd_mul = 1
        for j in range(1, n, 2):
            temp_even_sum = temp_even_sum + y[0][j] ** 2
            temp_even_mul = temp_even_mul * np.cos(20 * y[0][j] * np.pi /
                                                   (np.sqrt(j + 1)))
        if even_count != 0:
            temp_even = 2 * (4 * temp_even_sum - 2 *
                             temp_even_mul + 2) / even_count
        else:
            temp_even = 0
        for j in range(2, n, 2):
            temp_odd_sum = temp_odd_sum + y[0][j] ** 2
            temp_odd_mul = temp_odd_mul * np.cos(20 * y[0][j] * np.pi /
                                                 (np.sqrt(j + 1)))
        if odd_count != 0:
            temp_odd = 2 * (4 * temp_odd_sum - 2 *
                            temp_odd_mul + 2) / odd_count
        else:
            temp_odd = 0

        temp = np.max([0, 2 * (0.5 / N + eps) * np.abs(np.sin(2 * N * np.pi *
                                                              x[i][0]))])
        obj_func[i][0] = (x[i][0] + (0.5 / N + eps) * np.
                          abs(np.sin(2 * N * np.pi * x[i][0])) + temp_odd)
        obj_func[i][1] = (1 - x[i][0] + (0.5 / N + eps) * np.
                          abs(np.sin(2 * N * np.pi * x[i][0])) + temp_even)
        obj_func[i][2] = obj_func[i][0] + obj_func[i][1]
    return obj_func


def cal_obj_func_utf7(x):
    """
    This function calculates objective functions value
    -------------------------------------------------
    parameter
    x : list or array
     the value of decision variables
    return
    obj_func : array
            the value of objective functions
    --------------------------------------------------
    """
    x, even_count, odd_count = preprocess_two_obj(x)

    m = x.shape[0]
    n = x.shape[1]

    obj_func = np.zeros((m, 3))

    for i in range(0, m):
        y = np.zeros((1, n))
        for j in range(1, n):
            y[0][j] = x[i][j] - np.sin(6 * np.pi * x[i][0] +
                                       (j + 1) * np.pi / n)
        temp_even = 0
        temp_odd = 0
        for j in range(1, n, 2):
            temp_even = temp_even + y[0][j] ** 2
        if even_count != 0:
            temp_even = 2 * temp_even / even_count
        else:
            temp_even = 0
        for j in range(2, n, 2):
            temp_odd = temp_odd + y[0][j] ** 2
            temp_odd = 2 * temp_odd / odd_count
        else:
            temp_odd = 0
        obj_func[i][0] = np.power(x[i][0], 1 / 5) + temp_odd
        obj_func[i][1] = 1 - np.power(x[i][0], 1 / 5) + temp_even
        obj_func[i][2] = obj_func[i][0] + obj_func[i][1]
    return obj_func


def cal_obj_func_utf8(x):
    """
    This function calculates objective functions value
    -------------------------------------------------
    parameter
    x : list or array
     the value of decision variables which dimension is greater than one
    return
    obj_func : array
            the value of objective functions
    --------------------------------------------------
    """
    x, part1_count, part2_count, part3_count = preprocess_three_obj(x)

    m = x.shape[0]
    n = x.shape[1]

    obj_func = np.zeros((m, 4))

    for i in range(0, m):
        temp_part1 = 0
        temp_part2 = 0
        temp_part3 = 0
        for j in range(3, n, 3):
            temp_part1 = temp_part1 + (x[i][j] - 2 * x[i][1] *
                                       np.sin(2 * np.pi * x[i][0] +
                                              (j + 1) * np.pi / n)) ** 2
        if temp_part1 != 0:
            temp_part1 = 2 * temp_part1 / part1_count
        else:
            temp_part1 = 0
        for j in range(4, n, 3):
            temp_part2 = temp_part2 + (x[i][j] - 2 * x[i][1] *
                                       np.sin(2 * np.pi * x[i][0] +
                                              (j + 1) * np.pi / n)) ** 2
        if temp_part2 != 0:
            temp_part2 = 2 * temp_part2 / part2_count
        else:
            temp_part2 = 0
        for j in range(2, n, 3):
            temp_part3 = temp_part3 + (x[i][j] - 2 * x[i][1] *
                                       np.sin(2 * np.pi * x[i][0] +
                                              (j + 1) * np.pi / n)) ** 2
        if temp_part3 != 0:
            temp_part3 = 2 * temp_part3 / part3_count
        else:
            temp_part3 = 0
        obj_func[i][0] = (np.cos(0.5 * x[i][0] * np.pi) *
                          np.cos(0.5 * x[i][1] * np.pi) + temp_part1)
        obj_func[i][1] = (np.cos(0.5 * x[i][0] * np.pi) *
                          np.sin(0.5 * x[i][1] * np.pi) + temp_part2)
        obj_func[i][2] = np.sin(0.5 * x[i][0] * np.pi) + temp_part3
        obj_func[i][3] = obj_func[i][0] + obj_func[i][1] + obj_func[i][2]
    return obj_func


def cal_obj_func_utf9(x, eps=0.1):
    """
    This function calculates objective functions value
    -------------------------------------------------
    parameter
    x : list or array
     the value of decision variables which dimension is greater than one
    eps : float
       hyperparameter
    return
    obj_func : array
            the value of objective functions
    --------------------------------------------------
    """
    x, part1_count, part2_count, part3_count = preprocess_three_obj(x)

    m = x.shape[0]
    n = x.shape[1]

    obj_func = np.zeros((m, 4))

    for i in range(0, m):
        temp_part1 = 0
        temp_part2 = 0
        temp_part3 = 0
        for j in range(3, n, 3):
            temp_part1 = temp_part1 + (x[i][j] - 2 * x[i][1] *
                                       np.sin(2 * np.pi * x[i][0] +
                                              (j + 1) * np.pi / n)) ** 2
        if temp_part1 != 0:
            temp_part1 = 2 * temp_part1 / part1_count
        else:
            temp_part1 = 0
        for j in range(4, n, 3):
            temp_part2 = temp_part2 + (x[i][j] - 2 * x[i][1] *
                                       np.sin(2 * np.pi * x[i][0] +
                                              (j + 1) * np.pi / n)) ** 2
        if temp_part2 != 0:
            temp_part2 = 2 * temp_part2 / part2_count
        else:
            temp_part2 = 0
        for j in range(2, n, 3):
            temp_part3 = temp_part3 + (x[i][j] - 2 * x[i][1] *
                                       np.sin(2 * np.pi * x[i][0] +
                                              (j + 1) * np.pi / n)) ** 2
        if temp_part3 != 0:
            temp_part3 = 2 * temp_part3 / part3_count
        else:
            temp_part3 = 0
        temp = np.max([0, (1 + eps) * (1 - 4 * (2 * x[i][0] - 1) ** 2)])
        obj_func[i][0] = 0.5 * (temp + 2 * x[i][0]) * x[i][1] + temp_part1
        obj_func[i][1] = 0.5 * (temp - 2 * x[i][0] + 2) * x[i][1] + temp_part2
        obj_func[i][2] = 1 - x[i][1] + temp_part3
        obj_func[i][3] = obj_func[i][0] + obj_func[i][1] + obj_func[i][2]
    return obj_func


def cal_obj_func_utf10(x):
    """
    This function calculates objective functions value
    -------------------------------------------------
    parameter
    x : list or array
     the value of decision variables which dimension is greater than one
    return
    obj_func : array
            the value of objective functions
    --------------------------------------------------
    """
    x, part1_count, part2_count, part3_count = preprocess_three_obj(x)

    m = x.shape[0]
    n = x.shape[1]

    obj_func = np.zeros((m, 4))

    for i in range(0, m):
        y = np.zeros((1, n))
        for j in range(2, n):
            y[0][j] = x[i][j] - 2 * x[i][1] * np.sin(2 * np.pi * x[i][0] +
                                                     (j + 1) * np.pi / n)
        temp_part1 = 0
        temp_part2 = 0
        temp_part3 = 0
        for j in range(3, n, 3):
            temp_part1 = temp_part1 + 4 * (y[0][j] ** 2) -\
                         np.cos(8 * np.pi * y[0][j]) + 1
        if temp_part1 != 0:
            temp_part1 = 2 * temp_part1 / part1_count
        else:
            temp_part1 = 0
        for j in range(4, n, 3):
            temp_part2 = temp_part2 + 4 * (y[0][j] ** 2) -\
                         np.cos(8 * np.pi * y[0][j]) + 1
        if temp_part2 != 0:
            temp_part2 = 2 * temp_part2 / part2_count
        else:
            temp_part2 = 0
        for j in range(2, n, 3):
            temp_part3 = temp_part3 + 4 * (y[0][j] ** 2) -\
                         np.cos(8 * np.pi * y[0][j]) + 1
        if temp_part3 != 0:
            temp_part3 = 2 * temp_part3 / part3_count
        else:
            temp_part3 = 0
        obj_func[i][0] = (np.cos(0.5 * x[i][0] * np.pi) *
                          np.cos(0.5 * x[i][1] * np.pi) + temp_part1)
        obj_func[i][1] = (np.cos(0.5 * x[i][0] * np.pi) *
                          np.sin(0.5 * x[i][1] * np.pi) + temp_part2)
        obj_func[i][2] = np.sin(0.5 * x[i][0] * np.pi) + temp_part3
        obj_func[i][3] = obj_func[i][0] + obj_func[i][1] + obj_func[i][2]
    return obj_func
