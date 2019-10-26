# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     k-means
   email:         sjyttkl
   Author :       695492835@qq.com
   date：          2019/10/22
   Description :  
==================================================
"""
__author__ = 'songdongdong'


from collections import defaultdict
from random import  uniform
from math import sqrt


def point_avg(points):
    '''
    Accepts a list of points, each with the same number of dimensions.
    NB. points can have more dimensions than 2
    Returns a new points which is the center of all the points
    :param points:
    :return:
    '''
    dimensions = len(points[0])

    new_center = []

    for dimension in range(dimensions):
        dim_sum = 0
        for p in points:
            dim_sum += p[dimension]

        # average of each dimension
        new_center.append(dim_sum / float(len(points)))

    return new_center


def update_centers(date_set, assignments):
    '''
    Accepts a dataset and a list of assignments; the indexes of both lists correspond
    to each other.
    compute the center for each of the assigned groups.
    Reture 'k' centers where  is the number of unique assignments.
    :param date_set:
    :param assignments:
    :return:
    '''
    new_means = defaultdict(list)
    centers = []
    for assigment, point in zip(assignments, date_set):
        new_means[assigment].append(point)

    for points in new_means.values():
        centers.append(point_avg(points))

    return centers


def distance(a, b):
    dimensions = len(a)

    _sum = 0
    for dimension in range(dimensions):
        difference_seq = (a[dimension] - b[dimension]) ** 2
        _sum += difference_seq

    return sqrt(_sum)


def assign_points(data_points, centers):
    '''
    Given a data set and a list  of points between other points,
    assign each point to an index that corresponds to the index
    of the center point on its proximity to that point.
    Return a an array of indexes of the centers that correspond to
    an index in the data set; that is, if there are N points in data set
    the list we return will have N elements. Also If there ara Y points in centers
    there will be Y unique possible values within the returned list.
    :param data_points:
    :param centers:
    :return:
    '''
    assigments = []
    for point in data_points:
        shortest = float('Inf')
        shortest_index = 0
        for i in range(len(centers)):
            val = distance(point, centers[i])
            if val < shortest:
                shortest = val
                shortest_index = i
        assigments.append(shortest_index)

    return assigments


def generate_k(data_set, k):
    '''
    Given data set , which is an array of arrays,
    find the minimum and maximum foe each coordinate, a range.
    Generate k random points between the ranges
    Return an array of the random points within the ranges
    :param data_set:
    :param k:
    :return:
    '''
    centers = []
    dimensions = len(data_set[0])
    min_max = defaultdict(int)

    for point in data_set:
        for i in range(dimensions):
            val = point[i]
            min_key = 'min_%d' % i
            max_key = 'max_%d' % i
            if min_key not in min_max or val < min_max[min_key]:
                min_max[min_key] = val
            if max_key not in min_max or val > min_max[max_key]:
                min_max[max_key] = val

    for _k in range(k):
        rand_point = []
        for i in range(dimensions):
            min_val = min_max['min_%d' % i]
            max_val = min_max['max_%d' % i]

            rand_point.append(uniform(min_val, max_val))

        centers.append(rand_point)
    return centers


def k_means(dataset, k):
    k_points = generate_k(dataset, k)
    assignments = assign_points(dataset, k_points)
    old_assignments = None
    times = 0
    while assignments != old_assignments:
        times += 1
        print('times is :', times)
        new_centers = update_centers(dataset, assignments)
        old_assignments = assignments
        assignments = assign_points(dataset, new_centers)

    return (assignments, dataset)