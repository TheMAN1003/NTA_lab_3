import numpy as np

def extended_gcd(a, b):
    if b == 0:
        return a, 1, 0
    else:
        (d, x, y) = extended_gcd(b, a % b)
        return d, y, x - int(a / b) * y


def pivot(h: int, matrix: np.matrix, mod):

    for i in range(h, matrix.shape[0]):
        for j in range(h, matrix.shape[1]):
            d, _, _ = extended_gcd(matrix[i, j], mod)
            if d == 1:
                return i, j
    return -1, -1


def triangle(matrix, vector, mod):
    np.seterr(all='warn')

    xs = np.arange(0, matrix.shape[1])

    for k in range(0, matrix.shape[1]):

        i_max, j_max = pivot(k, matrix, mod)

        if i_max == -1 or j_max == -1:
            print("Mistake in matrix")
            return matrix, vector, xs, -1

        if j_max > k:
            matrix[:, [k, j_max]] = matrix[:, [j_max, k]]
            (xs[k], xs[j_max]) = (xs[j_max], xs[k])
        if i_max > k:
            matrix[[k, i_max]] = matrix[[i_max, k]]
            (vector[k], vector[i_max]) = (vector[i_max], vector[k])

        for i in range(k + 1, matrix.shape[0]):
            gcd, rev, _ = extended_gcd(matrix[k, k], mod)

            if gcd > 1:
                print("Not exist revers element")
                return matrix, vector, xs, -1

            f = (int(matrix[i, k]) * int(rev)) % mod

            for j in range(k, matrix.shape[1]):
                matrix[i, j] = (int(matrix[i, j]) - int(matrix[k, j]) * int(f)) % mod

            vector[i] = (int(vector[i]) - int(vector[k]) * int(f)) % mod
        print(k,matrix.shape[1])
    return matrix, vector, xs, 1


def reverse_steps(matrix, vector, mod):
    np.seterr(all='warn')

    solution = np.zeros(matrix.shape[1], dtype = np.int64)
    for i in reversed(range(0, matrix.shape[1])):
        number = 0

        for j in reversed(range(i + 1, matrix.shape[1])):
           # print(solution[j], matrix[i, j],)
            number += (int(solution[j]) * int(matrix[i, j])) % mod

        solution[i] = (vector[i] - number) % mod

        gcd, rev, _ = extended_gcd(matrix[i, i], mod)
        if gcd > 1:
            print("Not exist revers element")
            return -1

        solution[i] = (int(rev) * int(solution[i])) % mod

    return solution


def dotMat(matrix, vector, mod):
    result = []
    for row in matrix:
        row_result = 0
        for i in range(len(row)):
            row_result = (row_result + row[i] * vector[i]) % mod
        result.append(row_result)
    return np.array(result)

def Gauss(matrix, vector, mod):
    np.seterr(all='warn')

    matrix1 = np.copy(matrix)
    vector1 = list.copy(vector)

    matrix1, vector1, order, done = triangle(matrix1, vector1, mod)#todo variant when

    if done == -1:
        return vector1, -1


    midResult = reverse_steps(matrix1, vector1, mod)

    print(midResult)
    res = np.zeros(midResult.shape[0], dtype = np.int64)
    #print("sys GASUSS TRIAN", dotMat(matrix1, midResult, mod) - vector1)
    print("order=",order)
    for i in range(0, midResult.shape[0]):
        res[order[i]] = int(midResult[i])

    return res, 1
