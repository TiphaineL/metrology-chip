import numpy as np
from scipy import optimize
from problem.solution import solution
from curves.Bezier import Bezier_curve

def exponential_coefficient(x,y):
    a = (np.log(1. / y[0]) - np.log(1. / y[1])) / (x[1] - x[0])
    b = np.log(1. / y[1]) - a * x[0]
    return np.array([a,b])

def quadratic_coefficient(x,y):
    a = (y[1] - y[0]) / (x[1] ** 2 - x[0] ** 2)
    b = y[0] - a * x[0] ** 2
    return np.array([a, b])

class cost_TOF(object):

    def __init__(self):

        self.cost_low_high = np.array([.0001, .01])

        self.bend_low_high = np.array([70, 90])

        self.proximity_low_high = np.array([.03, 1])

        self.opd_std_low_high = np.array([.1e-4, .01])

        self.bend_radius_coefficients = exponential_coefficient(self.bend_low_high, self.cost_low_high)
        self.proximity_coefficients = exponential_coefficient(self.proximity_low_high, self.cost_low_high)
        self.opd_std_coefficients = quadratic_coefficient(self.opd_std_low_high, self.cost_low_high)

    def guess_points(self,variables):

        x_sep = 6
        y_sep = .1

        x0 = 3
        #y0 = 25

        [chip_length, y0] = variables

        point_1 = np.array([[x0,          0], \
                            [x0,          y0], \
                            [chip_length, y0 ]])

        point_2 = np.array([[x0 + x_sep,  0], \
                            [x0 + x_sep,  y0 - y_sep], \
                            [chip_length, y0 - y_sep]])

        point_3 = np.array([[x0 + 2*x_sep, 0], \
                            [x0 + 2*x_sep, y0 - 2*y_sep], \
                            [chip_length,  y0 - 2*y_sep]])

        point_4 = np.array([[x0 + 3*x_sep, 0], \
                            [x0 + 3*x_sep, y0 - 3*y_sep], \
                            [chip_length,  y0 - 3*y_sep]])

        return np.array([point_1,point_2,point_3,point_4])

    # def guess_points(self, variables):
    #
    #     x_sep = 6
    #     y_sep = .1
    #
    #     x0 = 3
    #     y0 = 25
    #
    #     chip_length = 30
    #
    #     [A1,B1] = variables
    #
    #     point_1 = np.array([[chip_length, x0], \
    #                         [A1,          x0], \
    #                         [B1,          y0],\
    #                         [chip_length, y0]])
    #
    #     return np.array([point_1])

    def function(self,variables):

        points = self.guess_points(variables)

        curve_1 = Bezier_curve(points[0])
        curve_2 = Bezier_curve(points[1])
        curve_3 = Bezier_curve(points[2])
        curve_4 = Bezier_curve(points[3])

        solution_ = solution(curve_1, curve_2, curve_3, curve_4)

        return 1/np.exp(self.bend_radius_coefficients [0] * solution_.min_bend_radius() \
                            +self.bend_radius_coefficients [1])


    def minimize(self,variables):

        bnds = ((0,50),(0,28))

        return optimize.minimize(self.function,variables,method='SLSQP',bounds=bnds)
