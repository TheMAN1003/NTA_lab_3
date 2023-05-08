from sympy.ntheory import factorint
from math import exp, log2, sqrt

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


