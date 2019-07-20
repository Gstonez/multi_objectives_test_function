#!/usr/bin env python
# -*-coding:utf-8-*-
# title        :zdt.py
# decription   :multi-objection test function zdt1~zdt4
# author       :Stone
# date         :20190702
#
# 考虑是否需要增加定义域判定
"""
This model is multi-objection test functions from zdt1 to zdt4

Input:
---------------------------------------------------------------
x : must be a list or array which is one-dimensional(individual)
    or two-dimensional(population)
 example :
 [1] : individual which decision variables is 1-d
 [1, 1] : individual which decision variables is 2-d
 [[1,1],[1,1]] :population which decision variables is 2-d
 ......

Output:
---------------------------------------------------------------
obj_func : array
        the objective functions value of x
"""
import numpy as np


def cal_obj_func_zdt1(x):
    '''
    This function calculates objective functions value
    -------------------------------------------------
    parameter
    x : list or array
     the value of decision variables
    return
    obj_func : array
            the value of objective functions
    --------------------------------------------------
    '''
    x = np.array(x)
    dim = x.ndim

    if dim == 1:
        x = np.array(x).reshape(-1, len(x))  # 单个体输入转换为2维，便于简化循环

    m = x.shape[0]  # 种群规模（输入样本个数）
    n = x.shape[1]  # 决策变量维度

    obj_func = np.zeros((m, 3))

    for i in range(0, m):
        temp = 0
        for j in range(1, n):
            temp = temp + x[i][j]
        if n > 1:
            temp = 1 + 9 * temp / (n - 1)
        else:
            temp = 1  # 避免决策变量维度为1，分母为零的情况
        obj_func[i][0] = x[i][0]
        obj_func[i][1] = 1 - np.sqrt(x[i][0] / temp)
        obj_func[i][2] = obj_func[i][0] + obj_func[i][1]
    return obj_func


def cal_obj_func_zdt2(x):
    '''
    This function calculates objective functions value
    -------------------------------------------------
    parameter
    x : list or array
     the value of decision variables
    return
    obj_func : array
            the value of objective functions
    --------------------------------------------------
    '''
    x = np.array(x)
    dim = x.ndim

    if dim == 1:
        x = np.array(x).reshape(-1, len(x))

    m = x.shape[0]
    n = x.shape[1]

    obj_func = np.zeros((m, 3))

    for i in range(0, m):
        temp = 0
        for j in range(1, n):
            temp = temp + x[i][j]
        if n > 1:
            temp = 1 + 9 * temp / (n - 1)
        else:
            temp = 1
        obj_func[i][0] = x[i][0]
        obj_func[i][1] = 1 - (x[i][0] / temp) ** 2
        obj_func[i][2] = obj_func[i][0] + obj_func[i][1]
    return obj_func


def cal_obj_func_zdt3(x):
    '''
    This function calculates objective functions value
    -------------------------------------------------
    parameter
    x : list or array
     the value of decision variables
    return
    obj_func : array
            the value of objective functions
    --------------------------------------------------
    '''
    x = np.array(x)
    dim = x.ndim

    if dim == 1:
        x = np.array(x).reshape(-1, len(x))

    m = x.shape[0]
    n = x.shape[1]

    obj_func = np.zeros((m, 3))

    for i in range(0, m):
        temp = 0
        for j in range(1, n):
            temp = temp + x[i][j]
        if n > 1:
            temp = 1 + 9 * temp / (n - 1)
        else:
            temp = 1
        obj_func[i][0] = x[i][0]
        obj_func[i][1] = 1 - (np.sqrt(x[i][0] / temp) - (x[i][0] / temp) *
                              np.sin(10 * np.pi * x[i][0]))
        obj_func[i][2] = obj_func[i][0] + obj_func[i][1]
    return obj_func


def cal_obj_func_zdt4(x):
    '''
    This function calculates objective functions value
    -------------------------------------------------
    parameter
    x : list or array
     the value of decision variables
    return
    obj_func : array
            the value of objective functions
    --------------------------------------------------
    '''
    x = np.array(x)
    dim = x.ndim

    if dim == 1:
        x = np.array(x).reshape(-1, len(x))

    m = x.shape[0]
    n = x.shape[1]

    obj_func = np.zeros((m, 3))

    for i in range(0, m):
        temp = 0
        for j in range(1, n):
            temp = temp + x[i][j] ** 2 - 10 * np.cos(4 * np.pi * x[i][j])
        temp = 1 + 10 * (n - 1) + temp
        obj_func[i][0] = x[i][0]
        obj_func[i][1] = 1 - np.sqrt(x[i][0] / temp)
        obj_func[i][2] = obj_func[i][0] + obj_func[i][1]
    return obj_func
