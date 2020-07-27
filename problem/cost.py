import numpy as np
from scipy import optimize
from problem.solution import solution
from curves.Bezier import Bezier_curve
from problem.guess_points import guess_points_fine_straight, guess_points_fine_90deg

def exponential_coefficient(x,y):
    a = (np.log(1. / y[0]) - np.log(1. / y[1])) / (x[1] - x[0])
    b = np.log(1. / y[1]) - a * x[0]
    return np.array([a,b])

def quadratic_coefficient(x,y):
    a = (y[1] - y[0]) / (x[1] ** 2 - x[0] ** 2)
    b = y[0] - a * x[0] ** 2
    return np.array([a, b])

class cost(object):

    def __init__(self,n):

        self.cost_low_high = np.array([.0001, .01])

        self.bend_low_high = np.array([70, 90])

        self.proximity_low_high = np.array([.03, 1])

        self.opd_std_low_high = np.array([.1e-4, .01])

        self.bend_radius_coefficients = exponential_coefficient(self.bend_low_high, self.cost_low_high)
        self.proximity_coefficients = exponential_coefficient(self.proximity_low_high, self.cost_low_high)
        self.opd_std_coefficients = quadratic_coefficient(self.opd_std_low_high, self.cost_low_high)

        self.design = n

    def function(self,variables):

        if self.design == 1:
            points = guess_points_fine_straight(variables)

        elif self.design == 2:
            points = guess_points_fine_90deg(variables)

        curve_1 = Bezier_curve(points[0])
        curve_2 = Bezier_curve(points[1])
        curve_3 = Bezier_curve(points[2])
        curve_4 = Bezier_curve(points[3])

        solution_ = solution(curve_1, curve_2, curve_3, curve_4)

        return 649500 / np.exp(self.proximity_coefficients[0] * solution_.min_proximity() \
                          + self.proximity_coefficients[1]) \
               + .4 / np.exp(self.bend_radius_coefficients[0] * solution_.min_bend_radius() \
                              + self.bend_radius_coefficients[1])\
                + 500 * (self.opd_std_coefficients[0] * solution_.standDev_OPD() ** 2 + self.opd_std_coefficients[1])

    def minimize(self,variables):

        if self.design ==1:
            bnds = ((0,30),(0,30),(0,30),(-10,10),(0,30),(0,30),(0,30),(0,30),(-10,10),(0,30),
                    (0,30),(0,30),(0,30),(-10,10),(0,30),(0,30),(0,30),(0,30),(-10,10),(0,30))

        elif self.design ==2:
            bnds = ((0,30),(0,30),(0,30),\
                    (0,30),(0,30),(0,30),(0,30),\
                    (0,30),(0,30),(0,30),(0,30),\
                    (0,30),(0,30),(0,30),(0,30), \
                    (-3,3),(-3,3),(-3,3),(-3,3), \
                    (0,30),(0,30),(0,30),(0,30) )

        return optimize.minimize(self.function,variables,method='SLSQP', bounds=bnds)