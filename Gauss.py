import copy

import numpy as np


def extended_gcd(a, b):
    if b == 0:
        return a, 1, 0
    else:
        (d, x, y) = extended_gcd(b, a % b)
        return d, y, x - int(a / b) * y


def pivot(h: int, matrix, mod):

    for i in range(h, len(matrix)):
        for j in range(h, len(matrix[0])):
            d, _, _ = extended_gcd(matrix[i][j], mod)
            if d == 1:
                return i, j
    return -1, -1


def swap_columns(matrix, col1, col2):
    for row in matrix:
        row[col1], row[col2] = row[col2], row[col1]
    return matrix


def swap_rows(matrix, row1, row2):
    matrix[row1], matrix[row2] = matrix[row2], matrix[row1]
    return matrix


def triangle(matrix, vector, mod):
    countCol = len(matrix[0])
    countRow = len(matrix)

    xs = np.arange(0, countCol)

    for k in range(0, countCol):

        i_max, j_max = pivot(k, matrix, mod)

        if i_max == -1 or j_max == -1:
            # print("Mistake in matrix")
            return matrix, vector, xs, -1

        if j_max > k:
            swap_columns(matrix, k, j_max)
            (xs[k], xs[j_max]) = (xs[j_max], xs[k])
        if i_max > k:
            swap_rows(matrix, k, i_max)
            (vector[k], vector[i_max]) = (vector[i_max], vector[k])

        for i in range(k + 1, countRow):
            gcd, rev, _ = extended_gcd(matrix[k][k], mod)

            if gcd > 1:
                # ("Not exist revers element")
                return matrix, vector, xs, -1

            f = ((matrix[i][k]) * (rev)) % mod

            for j in range(k, countCol):
                matrix[i][j] = ((matrix[i][j]) - (matrix[k][j]) * (f)) % mod

            vector[i] = ((vector[i]) - (vector[k]) * (f)) % mod

    return matrix, vector, xs, 1


def reverse_steps(matrix, vector, mod):

    countCol = len(matrix[0])
    countRow = len(matrix)

    solution = [0] * countCol
    for i in reversed(range(0, countCol)):
        number = 0

        for j in reversed(range(i + 1, countCol)):
            number += ((solution[j]) * (matrix[i][j])) % mod

        solution[i] = (vector[i] - number) % mod

        gcd, rev, _ = extended_gcd(matrix[i][i], mod)
        if gcd > 1:
            # ("Not exist revers element")
            return -1

        solution[i] = (rev * solution[i]) % mod

    return solution



def Gauss(matrix1, vector1, mod):
    matrix = copy.deepcopy(matrix1)
    vector = copy.deepcopy(vector1)
    # np.seterr(all='warn')
    #
    # matrix1 = np.copy(matrix)
    # vector1 = list.copy(vector)
    _, _, order, done = triangle(matrix, vector, mod)  # todo variant when

    if done == -1:
        return vector, -1

    midResult = reverse_steps(matrix, vector, mod)

    # print(midResult)
    res = [0]*len(midResult)
    # print("sys GASUSS TRIAN", dotMat(matrix1, midResult, mod) - vector1)
    # print("order=",order)
    for i in range(0, len(res)):
        res[order[i]] = midResult[i]

    return res, 1
