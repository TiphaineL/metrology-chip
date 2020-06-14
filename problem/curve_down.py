import numpy as np
from scipy import optimize
from problem.solution import solution
from curves.Bezier import Bezier_curve
from problem.cost import exponential_coefficient

class cost(object):

    def __init__(self):

        self.cost_low_high = np.array([.0001, .01])

        self.bend_low_high = np.array([70, 90])

        self.bend_radius_coefficients = exponential_coefficient(self.bend_low_high, self.cost_low_high)

    def guess_points(self,variables):

        [p2x,p3x,p4x] = variables

        h = 0.3
        point = np.array([[0, 0], \
                          [p2x, 0], \
                          [p3x, h], \
                          [p4x, h]])

        return point

    def function(self,variables):

        points = self.guess_points(variables)

        curve = Bezier_curve(points)

        solution_ = solution(curve)

        return 1/np.exp(self.bend_radius_coefficients [0] * solution_.min_bend_radius() \
                            +self.bend_radius_coefficients [1])

    def minimize(self,variables):
        bnds = ((-100, 100), (-100, 100), (0, 5.13))

        return optimize.minimize(self.function,variables,method='SLSQP', bounds=bnds)

h = 0.3
variables = np.array([0.5, 0.5, 1])
test = cost()

results = test.minimize(variables).x

print('results ',results)
points = test.guess_points(results)

curve = Bezier_curve(points)

solution_ = solution(curve)

solution_.plot_waveguides()

solution_.plot_bend_radii()

print('Points ',points)

print('bend radius ',solution_.min_bend_radius())
