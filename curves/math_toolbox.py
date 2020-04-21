import math
import numpy as np

def vector_length(vector):
    return np.sqrt(sum(element ** 2 for element in vector))

def cross_product(vector1, vector2):

    if len(vector1) & len(vector2) == 3:

        x = vector1[1] * vector2[2] - vector1[2] * vector2[1]
        y = vector1[2] * vector2[0] - vector1[0] * vector2[2]
        z = vector1[0] * vector2[1] - vector1[1] * vector2[0]

    elif len(vector1) & len(vector2) == 2:
        x = vector1[1] * 0 - 0 * vector2[1]
        y = 0 * vector2[0] - vector1[0] * 0
        z = vector1[0] * vector2[1] - vector1[1] * vector2[0]

    return np.array([x, y, z])

def binomial(n, k):
    return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))

def transpose(array):
    return np.hsplit(array, array.shape[1])