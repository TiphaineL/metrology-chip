from curves.Bezier import Bezier_curve
import numpy as np
from problem_variables.initial_conditions import *
from curves.math_toolbox import vector_length, transpose
import matplotlib.pyplot as plt
from problem.cost import cost

def distance(curve1,curve2):

    distance_vector = curve1.function() - transpose(curve2.function())
    all_distances = np.array([ vector_length(vector) for vector in distance_vector ])
    distance = all_distances.min(axis=1)

    return distance

class solution:

    def __init__(self, *curves):
        self.number_of_guides = len(curves)

        self.waveguides = []
        for curve in curves:
            self.waveguides.append(curve)

    def plot_waveguides(self):
        for waveguide in self.waveguides:
            waveguide.plot()

    def standDev_OPD(self):
        OPD = [waveguide.arc_length() for waveguide in self.waveguides]
        return np.std(OPD)

    def plot_bend_radii(self):
        plt.figure()
        i = 0
        for waveguide in self.waveguides:
            i += 1
            plt.plot(waveguide.bend_radius(), label='waveguide ' + str(i))
            plt.xlabel('Chip length (arbitrary unit)')
            plt.ylabel('Bend radius')
            plt.legend(frameon=False)
            plt.show()

    def min_bend_radius(self):
        min_bend_radius = [waveguide.min_bend_radius() for waveguide in self.waveguides]
        return min(min_bend_radius)

    def plot_distances(self):

        plt.figure()

        for i in range(self.number_of_guides):
            for j in range(i + 1, self.number_of_guides):

                proximity = distance(self.waveguides[i], self.waveguides[j])
                label = 'waveguide ' + str(i + 1) + ' and waveguide ' + str(j + 1)
                plt.plot(proximity, label=label)
        #plt.plot(constraint, label='Minimum proximity: 30$\mu$m')
        plt.legend(frameon=False)
        plt.xlabel('Chip length (arbitrary unit)')
        plt.ylabel('Minimum proximity ($\mu$m)')
        # plt.title('Proximity between each waveguide pair')
        plt.show()

    def min_proximity(self):
        min_proximity = []
        for i in range(self.number_of_guides):
            for j in range(i + 1, self.number_of_guides):

                proximity = distance(self.waveguides[i], self.waveguides[j])
                min_proximity.append(min(proximity))
        return min(min_proximity)

point_1 = np.array([[10, h1],\
[x-7*w + h4*np.tan(theta_4*np.pi/180), h4],\
                    [x-7*w, 0]])

point_2 = np.array([[10, h2],\
                    [x-3*w + h3*np.tan(theta_3*np.pi/180), h3],
                    [x-3*w, 0]])

point_3 = np.array([[10, h3],\
                    [x-2*w + h2*np.tan(theta_2*np.pi/180), h2],\
                    [x-2*w, 0]])

point_4 = np.array([[10, h4], \
                    [x + h1 * np.tan(theta_1 * np.pi / 180), h1], \
                    [x, 0]])

curve_1 = Bezier_curve(point_1)
curve_2 = Bezier_curve(point_2)
curve_3 = Bezier_curve(point_3)
curve_4 = Bezier_curve(point_4)

solution_ = solution(curve_1,curve_2,curve_3,curve_4)
solution_.plot_waveguides()
solution_.plot_distances()
solution_.plot_bend_radii()
print(solution_.standDev_OPD())
print(solution_.min_bend_radius())
print(solution_.min_proximity())

cost_F = cost(solution_,[20,100],[.3,.7],[1e-3,.1e-3])



