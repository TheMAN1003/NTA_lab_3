from sympy.ntheory import factorint
from math import exp, log2, sqrt, log10
import numpy as np
import Gauss as gs
import random
import time
import multiprocessing

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
    return 3.38 * exp(0.5 * sqrt(log2(n) * log2(log2(n))))


def buildBase(n):
    nprime = 0
    primelist = []
    L = nSize(n)
    with open('primes.txt') as file:
        for line in file:
            p = int(line)
            if p < L:
                primelist.append(p)
            else:
                return primelist
    return primelist

def binarySearch(x, inList):
    left = 0
    right = len(inList) - 1

    while left <= right:
        mid = int((left + right) / 2)
        if inList[mid] == x:
            return 1, mid
        elif inList[mid] < x:
            left = mid + 1
        else:
            right = mid - 1
    return 0, -1


def checkSmooth(s, base):
    p_i = factorint(s)
    for i in p_i:
        if binarySearch(p_i, inList=base) == 0:
            return 0, 0
    return 1, p_i

def generate_random_sequence(n):
    sequence = list(range(1, n + 1))
    random_sequence = random.sample(sequence, n)
    return random_sequence

def dotMat(matrix, vector, mod):
    result = []
    for row in matrix:
        row_result = 0
        for i in range(len(row)):
            row_result = (row_result + row[i] * vector[i]) % mod
        result.append(row_result)
    return np.array(result)


def findVector(alpha, k ,n , base):
    p_i = factorint(powMod(alpha, k, n))

    vector = [0] * len(base)
    push = 1
    for p, power in p_i.items():
        exist, indexP = binarySearch(p, base)
        if exist == 1:
            vector[indexP] = power
        else:
            push = 0
            return vector, -1
    if push == 1:
        return vector, k

def findSystem(alpha, base, n):

    system = []
    B = []
    c = int(200*log10(n)) + 1
    end = c + len(base)
    endpar = c + len(base)
    i = 0
    numproc = 10
    pool = multiprocessing.Pool(processes=numproc)
    while(True):

        tasks = []
        start_time = time.time()


        while len(system) <= end:
            tasks = []
            i = 0
            while i < endpar:
                k = random.randint(0, n - 1)
                tasks.append(pool.apply_async(findVector, (alpha, k, n, base)))
                i += 1


            for task in tasks:
                tmp = task.get()
                if tmp is not None:
                    if tmp[1] != -1:
                        system.append(tmp[0])
                        B.append(tmp[1])

        end_time = time.time()

        execution_time = end_time - start_time
        print("Час виконання: ", execution_time, "секунд")
        print("syssize", len(system))

        #pool.close()

        res, done = gs.Gauss(system, B, n - 1)

        if done == -1:
            end += c
        else:
            return res


def findLogBeta(alpha, beta, base, n, smallLog):
    while(True):
        l = random.randint(0, n-1)
        tmp = (beta * powMod(alpha, l, n)) % n
        p_i = factorint(tmp)
        smooth = 1
        num = 0
        for p, power in p_i.items():
            exist, indexP = binarySearch(p, base)
            if exist == 1:
                num = (num + smallLog[indexP]*power) % (n - 1)
            else:
                smooth = 0
                break
        if smooth == 1:
            return (num - l) % (n - 1)

def indexCalculus(alpha, beta, n):
    base = buildBase(n)
    print(len(base))
    print("BUILDED BASE")
    smallLogs = findSystem(alpha, base, n)
    for i in range(len(smallLogs)):
        if powMod(alpha, int(smallLogs[i]), n) != base[i]:
            print(smallLogs[i],int(smallLogs[i]))
            print("PROBLEM WITH SMALL LOGS")
            return -1
    print("FOUND SMALL LOGS")
    return findLogBeta(alpha, beta, base, n, smallLogs)





