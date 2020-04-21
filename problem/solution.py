from curves.Bezier import Bezier_curve
import numpy as np
from problem_variables.initial_conditions import *
from curves.math_toolbox import vector_length, transpose
import matplotlib.pyplot as plt

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

    def print_arclength(self):
        for waveguide in self.waveguides:
            print(waveguide.arc_length())

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