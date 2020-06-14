import numpy as np
from problem_variables.initial_conditions import D_output_angles_rad, D_output_non_redundant_array, C_pitch_array
from curves.Bezier import Bezier_curve
import matplotlib.pyplot as plt

def guess_points_fine_90deg(variables):

    [start_x,start_y, chip_length, y2, x3, y3, z3, x4] = variables

    x1 = start_x + D_output_non_redundant_array
    y1 = 0 * np.ones(4)
    z1 = 0 * np.ones(4)

    x2 = x1 + y2*np.tan(D_output_angles_rad)
    y2 = y2 * np.ones(4)
    z2 = 0 * np.ones(4)

    x3 = x3 * np.ones(4)
    y3 = y3 * np.ones(4)
    z3 = z3 * np.ones(4)

    x5 = chip_length * np.ones(4)
    y5 = start_y - C_pitch_array
    z5 = 0 * np.ones(4)

    x4 = x4 * np.ones(4)
    y4 = y5
    z4 = 0 * np.ones(4)

    point = np.array([ [x1, y1, z1],
                       [x2, y2, z2],
                       [x3, y3, z3],
                       [x4, y4, z4],
                       [x5, y5, z5] ])

    return point

variables = np.array([0, 30, 50, 10, 25, 19, 2, 42])

#points = guess_points_fine_90deg(var).transpose()

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#for waveguide in points:
#    waveguide_points = waveguide.transpose()
#    print(waveguide_points)
#    print()
#    curve = Bezier_curve(waveguide_points)

#    ax.plot(curve.function()[0], curve.function()[1], curve.function()[2])
#ax.set_zlim([-30,30])
#ax.set_xlabel('X (mm)')
#ax.set_ylabel('Y (mm)')
#ax.set_zlabel('Z (mm)')
