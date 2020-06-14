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

class cost(object):

    def __init__(self):

        self.cost_low_high = np.array([.0001, .01])

        self.bend_low_high = np.array([70, 90])

        self.proximity_low_high = np.array([.03, 1])

        self.opd_std_low_high = np.array([.1e-4, .01])

        self.bend_radius_coefficients = exponential_coefficient(self.bend_low_high, self.cost_low_high)
        self.proximity_coefficients = exponential_coefficient(self.proximity_low_high, self.cost_low_high)
        self.opd_std_coefficients = quadratic_coefficient(self.opd_std_low_high, self.cost_low_high)

        self.w = 2.5 * 6e-3

        theta_1 = 1.72
        theta_2 = 0
        theta_3 = -.43
        theta_4 = -1.29

        self.angles = np.array([theta_1, theta_2, theta_3, theta_4])

    def guess_points(self,variables):

        x0 = 50
        y0 = 28

        w = .2
        y = 28 - 8.33 - 7 * w

        [A0,B0,C0,D0,E0,A1,B1,C1,D1,E1,A2,B2,C2,D2,E2,A3,B3,C3,D3,E3] = variables

        point_1 = np.array([[x0, y0, 0], \
                            [A0, y0, 0], \
                            [B0, C0, D0], \
                            [E0, y + 7 * w + E0 * np.tan(self.angles[0] * np.pi / 180), 0], \
                            [0,  y + 7 * w, 0]])

        point_2 = np.array([[x0, y0 - 0.1, 0], \
                            [A1, y0 - 0.1, 0], \
                            [B1, C1, D1], \
                            [E1, y + 3 * w + E1 * np.tan(self.angles[1] * np.pi / 180), 0],
                            [0,  y + 3 * w, 0]])

        point_3 = np.array([[x0, y0 - 0.2, 0], \
                            [A2, y0 - 0.2, 0], \
                            [B2, C2, D2], \
                            [E2, y + 2 * w + E2 * np.tan(self.angles[2] * np.pi / 180), 0], \
                            [0,  y + 2 * w, 0]])

        point_4 = np.array([[x0, y0 - 0.3, 0], \
                            [A3, y0 - 0.3, 0], \
                            [B3, C3, D3], \
                            [E3, y + E3 * np.tan(self.angles[3] * np.pi/180), 0],\
                            [0,  y + 0, 0]])

        return np.array([point_1, point_2, point_3, point_4])

    def function(self,variables):

        points = self.guess_points(variables)

        curve_1 = Bezier_curve(points[0])
        curve_2 = Bezier_curve(points[1])
        curve_3 = Bezier_curve(points[2])
        curve_4 = Bezier_curve(points[3])

        solution_ = solution(curve_1, curve_2, curve_3, curve_4)

        return 40.0 / np.exp(self.proximity_coefficients[0] * solution_.min_proximity() \
                           + self.proximity_coefficients[1]) \
                + 28000.0 * (self.opd_std_coefficients[0] * solution_.standDev_OPD() ** 2 + self.opd_std_coefficients[1]) \
                + 1/np.exp(self.bend_radius_coefficients [0] * solution_.min_bend_radius() \
                            +self.bend_radius_coefficients [1])


    def minimize(self,variables):

        bnds = ((0,30),(0,30),(0,30),(-10,10),(0,30),(0,30),(0,30),(0,30),(-10,10),(0,30),
                (0,30),(0,30),(0,30),(-10,10),(0,30),(0,30),(0,30),(0,30),(-10,10),(0,30))

        return optimize.minimize(self.function,variables,method='SLSQP',bounds=bnds)
