from sympy.ntheory import factorint
from math import exp, log2, sqrt
import numpy as np
import Gauss as gs

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


def findLog(l, divlist, loglist, mod):
    sum = 0
    for i in range(loglist):
        sum += (divlist[i] * loglist[i] - l) % mod
    return sum

# matrix = [[5, 7, 11], [12, 9, 1], [4, 13, 17]]
# vector = [4, 5, 1]


# print(gs.Gauss(matrix, vector, 36))

a = 1
b = 1
mod = 10
n = mod - 1
# x1  x2 x3 x4
# 1   1  0  0  13
# 0   1  1  0  19
# 1   0  1  0  22
# 1   0  0  1  20


m1 = [[1,  1, 0,  0],
[0,  0, 1,  0],
[0,  0, 1, 35],
[0,  0, 2,  0],
[0,  0, 0,  1]]

matrix = np.matrix([[1, 1, 0, 0],
                    [0, 1, 1, 0],
                    [1, 0, 0, 1],
                    [1, 0, 1, 0],
                    [0, 1, 0, 0]])

matrix[:, [1, 2]] = matrix[:, [2, 1]]

vector = np.array([13, 19, 20, 22, 23])
#
# (vector[0], vector[1]) = (vector[1], vector[0])
# print(vector)
print(gs.triangle(matrix, vector, 36))

print(gs.extended_gcd(35,36))


