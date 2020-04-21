import numpy as np
from scipy import optimize
from problem.solution import solution
from curves.Bezier import Bezier_curve

def exponential_coefficient(x,y):
    a = (np.log(1. / y[1]) - np.log(1. / y[0])) / (x[1] - x[0])
    b = np.log(1. / y[0]) - a * x[0]
    return np.array([a,b])

def quadratic_coefficient(x,y):
    a = (y[1] - y[0]) / (x[1] ** 2 - x[0] ** 2)
    b = y[0] - a * x[0] ** 2
    return np.array([a, b])

class cost(object):

    def __init__(self):

        self.cost_low_high = np.array([.0001, .01])

        high_bend = 20
        low_bend = 100

        low_proximity = .3
        high_proximity = .7

        low_opd_std = .1e-3
        high_opd_std = .5e-3

        self.bend_radius_coefficients = exponential_coefficient(self.cost_low_high, [low_bend, high_bend])
        self.proximity_coefficients = exponential_coefficient(self.cost_low_high, [low_proximity, high_proximity])
        self.opd_std_coefficients = quadratic_coefficient(self.cost_low_high, [high_opd_std, low_opd_std])

        #h1 = 4
        #h2 = 4 - 0.1
        #h3 = 4 - 0.2
        #h4 = 4 - 0.3

        #self.variables = np.array([h1, h2, h3, h4,x])

        #self.x = 8

        self.w = .02

        theta_1 = -10
        theta_2 = -.1
        theta_3 = 0
        theta_4 = 10

        self.angles = np.array([theta_1, theta_2, theta_3, theta_4])

    def guess_points(self,variables):

        x = variables[4]
        #w = variables[5]
        point_1 = np.array([[15, variables[0]], \
                            [x - 7 * self.w + variables[0] * np.tan(self.angles[0] * np.pi / 180), variables[0]], \
                            [x - 7 * self.w, 0]])

        point_2 = np.array([[15, variables[1]], \
                            [x - 3 * self.w + variables[1] * np.tan(self.angles[1] * np.pi / 180), variables[1]],
                            [x - 3 * self.w, 0]])

        point_3 = np.array([[15, variables[2]], \
                            [x - 2 * self.w + variables[2] * np.tan(self.angles[2] * np.pi / 180), variables[2]], \
                            [x - 2 * self.w, 0]])

        point_4 = np.array([[15, variables[3]], \
                            [x + variables[3] * np.tan(self.angles[3] * np.pi / 180), variables[3]], \
                            [x, 0]])

        return np.array([point_1, point_2, point_3, point_4])

    def function(self,variables):

        points = self.guess_points(variables)

        curve_1 = Bezier_curve(points[0])
        curve_2 = Bezier_curve(points[1])
        curve_3 = Bezier_curve(points[2])
        curve_4 = Bezier_curve(points[3])

        solution_ = solution(curve_1, curve_2, curve_3, curve_4)

        #return #5.8 / np.exp(self.bend_radius_coefficients [0] * solution_.min_bend_radius() \
               #             +self.bend_radius_coefficients [1]) + \
               #2.0 / np.exp(self.proximity_coefficients[0] * solution_.min_proximity() \
               #             + self.proximity_coefficients[1]) + \
        #return self.opd_std_coefficients[0] * solution_.standDev_OPD() ** 2 + self.opd_std_coefficients[1]
        return solution_.standDev_OPD()

    def minimize(self,variables):
        bnds = ((10, 25), (10, 25), (10, 25), (10,25), (5, 25))#, (-.01, .01))
        return optimize.minimize(self.function,variables,method='SLSQP',bounds=bnds)#np.array([[3,5],[3,5],[3,5],[3,5]]))


#sol = scipy.optimize.minimize(cost,\
#                               variables_0,\
#                               method='Powell',bounds= bnds)