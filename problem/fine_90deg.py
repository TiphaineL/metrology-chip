import numpy as np
from problem_variables.initial_conditions import fine_output_angles_rad, fine_output_non_redundant_array, C_pitch_array
from curves.Bezier import Bezier_curve
import matplotlib.pyplot as plt


def guess_points_fine_90deg(variables):

    [start_x,start_y, chip_length, Y2, X3, Y3, Z3, X4] = variables

    points = []
    for i in range(4):

        x1 = start_x + fine_output_non_redundant_array[i]
        y1 = 0
        z1 = 0

        x2 = x1 + Y2[i] * np.tan(fine_output_angles_rad[i])
        y2 = Y2[i]
        z2 = 0

        x3 = X3[i]
        y3 = Y3[i]
        z3 = Z3[i]

        x5 = chip_length
        y5 = start_y - C_pitch_array[i]
        z5 = 0

        x4 = X4[i]
        y4 = y5
        z4 = 0

        points.append( np.array([ [x1, y1, z1],
                                  [x2, y2, z2],
                                  #[x3, y3, z3],
                                  [x4, y4, z4],
                                  [x5, y5, z5] ]) )


    return points

variables = np.array( [0, 30, 50, [10, 10, 10 ,10], [25,25,25,25], [19,19,19,19], [2,2,2,2], [42,42,42,42] ])

points = guess_points_fine_90deg(variables)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for waveguide in points:
    print(waveguide)
    print()
    curve = Bezier_curve(waveguide)

    ax.plot(curve.function()[0], curve.function()[1], curve.function()[2])
ax.set_zlim([-30,30])
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
