import numpy as np
import matplotlib.pyplot as plt
from curves.math_toolbox import vector_length, binomial, cross_product

def bezier_function_1D(curve):

    bezier_function = 0
    dummy = np.arange(start=0, stop=1, step=1e-3)

    for n in range(curve.order + 1):
        bezier_order_n = binomial(curve.order, n) * (1 - dummy) ** (curve.order - n) \
                           * dummy ** (n) * curve.points[n]
        bezier_function += bezier_order_n
    return bezier_function

def bezier_function(curve):

    bezier_function = []

    for d in range(curve.dimension):

        points_in_1d = curve.points[:,d]
        bezier_function.append(bezier_function_1D(Bezier_curve(points_in_1d)))

    return np.array(bezier_function)

def points_derivative(curve):

    derivated_points = []
    for i in range(curve.order):
        derivated_points.append((curve.order) * (curve.points[i + 1] - curve.points[i]))

    return np.array(derivated_points)

def bezier_derivative_curve(curve):
    return Bezier_curve(points_derivative(curve))

def bezier_arc_length(curve):

    bezier_derivative = bezier_derivative_curve(curve)

    curve_derivative = bezier_derivative.function()

    L = vector_length(curve_derivative)

    return sum(L)/len(L)

def bezier_curvature(curve):

    derivative_1st_order = bezier_derivative_curve(curve)
    derivative_2nd_order = bezier_derivative_curve(derivative_1st_order)

    return vector_length(cross_product(derivative_1st_order.function(), derivative_2nd_order.function())) \
           / (vector_length(derivative_1st_order.function())) ** 3

class Bezier_curve:

    def __init__(self,control_points):
        self.points = control_points
        self.order = control_points.shape[0] - 1

        if len(control_points.shape) > 1:
            self.dimension = control_points.shape[1]
        else:
            self.dimension = 1

    def function(self):
        return bezier_function(self)

    def arc_length(self):
        return bezier_arc_length(self)

    def curvature(self):
        return bezier_curvature(self)

    def bend_radius(self):
        return 1.0/self.curvature()

    def min_bend_radius(self):
        return min(self.bend_radius())

    def plot(self):

        if self.dimension == 1:
            plt.plot(self.function())

        elif self.dimension == 2:
            plt.plot(self.function()[0], self.function()[1])

        elif self.dimension == 3:

            fig = plt.figure()

            for i in range(3):
                ax = fig.add_subplot(111, projection='3d')
                ax.plot(self.function()[0], self.function()[1], self.function()[2])


        else:
            print('No plotting for dimension > 3')



