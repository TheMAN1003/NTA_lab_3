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


def triangle(matrix: np.matrix, vector: np.array, mod: int):

    xs = np.arange(0, matrix.shape[1])

    for k in range(matrix.shape[1]):

        i_max, j_max = pivot(k, matrix, mod)

        if i_max == -1 or j_max == -1:
            print("Mistake in matrix")
            return -1

        matrix[[k, i_max]] = matrix[[i_max, k]]

        if j_max > k:
            matrix[:, [k, j_max]] = matrix[:, [j_max, k]]
            (xs[k], xs[j_max]) = (xs[j_max], xs[k])

        vector[k], vector[i_max] = vector[i_max], vector[k]

        for i in range(k + 1, matrix.shape[0]):
            gcd, rev, _ = extended_gcd(matrix[k, k], mod)

            if gcd > 1:
                print("Not exist revers element")
                return -1

            f = (matrix[i, k] * rev) % mod

            for j in range(k + 1, matrix.shape[1]):
                matrix[i, j] = (matrix[i, j] - matrix[k, j] * f) % mod

            vector[i] = (vector[i] - vector[k] * f) % mod

            matrix[i, k] = 0 #TODO why???
        print(matrix)
    print(xs)
    print(vector)
    return matrix, vector


def reverse_steps(matrix, vector, mod):
    solution = np.zeros_like(vector)
    for i in reversed(range(len(matrix))):
        number = 0

        for j in reversed(range(len(matrix))):
            number += solution[j]

        gcd, rev, _ = extended_gcd(vector[i] - number, matrix[i][i])  # TODO BUG(BAG)
        if gcd > 1:
            print("Not exist revers element")
            return -1
        solution[i] = rev % mod
    return solution


def Gauss(matrix, vector, mod):
    matrix, vector = triangle(matrix, vector, mod)

    result = reverse_steps(matrix, vector, mod)
    return result
