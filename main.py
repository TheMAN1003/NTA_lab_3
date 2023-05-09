from sympy.ntheory import factorint
from math import exp, log2, sqrt
import numpy as np

def extended_gcd(a, b):
    if b == 0:
        return a, 1, 0
    else:
        (d, x, y) = extended_gcd(b, a % b)
        return d, y, x - int(a / b) * y

def powMod(x, pow, mod):
    C = 1
    if pow == 0:
        return C
    powB = bin(pow)[2:]
    for i in range(len(powB)):
        if powB[i] == '1':
            C = (C * x) % mod
        if i != len(powB) - 1:
            C = (C * C) % mod
    return C
def nSize(n):
    return 3.38*exp(0.5*sqrt(log2(n)*log2(log2(n))))

def buildBase(n):
    nprime = 0
    primelist = []
    primelistsize = nSize(n)
    with open('primes.txt') as file:
        for line in file:
            if nprime < primelistsize:
                primelist.append(line.rstrip())
                nprime += 1
    return primelist

def checkSmooth(alpha, base, mod, n):
    listk = []
    matrix = []
    for k in range(0, n):
        number = powMod(alpha, k, mod)
        dividers = factorint(number)
        for prime in dividers:
            if prime not in base:
                continue
        listk.append(k)
        if len(listk) > len(base) + 10:
            break
        matrix.append(dividers.values())
    return matrix, listk

def pivot(h, matrix):
    max = 0
    counter = 0
    for i in range(h, len(matrix)):
        if abs(matrix[i][h]) > max:
            max = abs(matrix[i][h])
            counter = i
    return counter

def triangle(matrix, vector, mod):
    vector = np.array(vector)
    matrix = np.array(matrix)
    minlen = min(len(matrix), len(matrix[0]))
    for k in range(minlen):
        i_max = pivot(k, matrix)
        if matrix[i_max][k] == 0:
            print("Mistake in matrix")
            return
        matrix[[k, i_max]] = matrix[[i_max, k]]
        vector[k], vector[i_max] = vector[i_max], vector[k]
        for i in range(k+1, len(matrix)):
            f = (matrix[i][k] * extended_gcd(matrix[k][k], mod)[1]) % mod
            for j in range(k+1, len(matrix[0])):
                matrix[i][j] = (matrix[i][j] - matrix[k][j]*f) % mod
                vector[i] = (vector[i] - vector[k]*f) % mod
            matrix[i][k] = 0
    return matrix, vector

# def Gauss(matrix, vector, mod):
#     solution = []
#     for i in reversed(range(len(matrix))):


matrix = [[5, 7, 11], [12, 9, 1], [4, 13, 17]]
vector = [4, 5, 1]

print(triangle(matrix, vector, 37))


a = 1
b = 1
mod = 10
n = mod - 1

