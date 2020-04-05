import numpy as np

def exponential_coefficient(x,y):
    a = (np.log(1. / y[1]) - np.log(1. / y[0])) / (x[1] - x[0])
    b = np.log(1. / y[0]) - a * x[0]
    return np.array([a,b])

def quadratic_coefficient(x,y):
    a = (y[1] - y[0]) / (x[1] ** 2 - x[0] ** 2)
    b = y[0] - a * x[0] ** 2
    return np.array([a, b])

class cost:

    def __init__(self, solution, bend_radius_high_low_values, proximity_high_low_values, opd_std_high_low_values):

        self.solution = solution
        self.cost_high_low = np.array([1, .1])

        self.bend_radius_coefficients = exponential_coefficient(self.cost_high_low, bend_radius_high_low_values)
        self.proximity_coefficients = exponential_coefficient(self.cost_high_low, proximity_high_low_values)
        self.opd_std_coefficients = quadratic_coefficient(self.cost_high_low,opd_std_high_low_values)

    def function(self):

        return 5.8 / np.exp(self.bend_radius_coefficients [0] * self.solution.min_bend_radius() \
                            +self.bend_radius_coefficients [1]) + \
               2.0 / np.exp(self.proximity_coefficients * self.solution.min_proximity() \
                            + self.proximity_coefficients[1]) + \
               0.3 * (self.opd_std_coefficients[0] * self.solution.standDev_OPD() ** 2 + self.opd_std_coefficients[1])


