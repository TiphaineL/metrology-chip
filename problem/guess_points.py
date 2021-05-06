from problem_variables.initial_conditions import fine_output_angles_rad, fine_output_non_redundant_array, sidestep, \
                                                 C_pitch_array, TOF_pitch, C_input_pitch, chip_length
import numpy as np

def guess_points_fine_straight(variables):

    angles = fine_output_angles_rad

    x0 = 50
    y0 = 28

    w = fine_output_non_redundant_array
    y = 28 - sidestep

    [A0, B0, C0, D0, E0, A1, B1, C1, D1, E1, A2, B2, C2, D2, E2, A3, B3, C3, D3, E3] = variables

    point_1 = np.array([[x0, y0, 0], \
                        [A0, y0, 0], \
                        [B0, C0, D0], \
                        [E0, y - w[0] + E0 * np.tan(angles[0]), 0], \
                        [0, y - w[0], 0]])

    point_2 = np.array([[x0, y0 - 0.1, 0], \
                        [A1, y0 - 0.1, 0], \
                        [B1, C1, D1], \
                        [E1, y - w[1] + E1 * np.tan(angles[1]), 0],
                        [0, y - w[1], 0]])

    point_3 = np.array([[x0, y0 - 0.2, 0], \
                        [A2, y0 - 0.2, 0], \
                        [B2, C2, D2], \
                        [E2, y - w[2] + E2 * np.tan(angles[2]), 0], \
                        [0, y -w[2], 0]])

    point_4 = np.array([[x0, y0 - 0.3, 0], \
                        [A3, y0 - 0.3, 0], \
                        [B3, C3, D3], \
                        [E3, y - w[3] + E3 * np.tan(angles[3]), 0], \
                        [0, y - w[3] + 0, 0]])

    return np.array([point_1, point_2, point_3, point_4])

def guess_points_fine_90deg(variables):

    [start_x,start_y, chip_length, Y2_1, Y2_2, Y2_3, Y2_4, X3_1, X3_2, X3_3, X3_4, Y3_1,Y3_2, Y3_3, Y3_4,\
     Z3_1, Z3_2, Z3_3, Z3_4, X4_1, X4_2, X4_3, X4_4] = variables

    Y2 = np.array([Y2_1, Y2_2, Y2_3, Y2_4])
    X3 = np.array([X3_1, X3_2, X3_3, X3_4])
    Y3 = np.array([Y3_1, Y3_2, Y3_3, Y3_4])
    Z3 = np.array([Z3_1, Z3_2, Z3_3, Z3_4])
    X4 = np.array([X4_1, X4_2, X4_3, X4_4])

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
                                  [x3, y3, z3],
                                  [x4, y4, z4],
                                  [x5, y5, z5] ]) )


    return points

def guess_points_TOF_straight(variables):

    x0 = 0
    y0 = 3 + 3*TOF_pitch

    y4 = x0 + 3*TOF_pitch + sidestep + 3*C_input_pitch
    x4 = chip_length

    [X1_1, X1_2, X1_3, X1_4, X2_1, X2_2, X2_3, X2_4] = variables

    X1 = np.array([X1_1, X1_2, X1_3, X1_4])
    X2 = np.array([X2_1, X2_2, X2_3, X2_4])

    points = []
    for i in range(4):

        x1 = x0
        y1 = y0  - i*TOF_pitch
        z1 = 0

        x2 = X1[i]
        y2 = y1
        z2 = 0

        x4 = x4
        y4 = y4 - i*C_input_pitch
        z4 = 0

        x3 = X2[i]
        y3 = y4
        z3 = 0

        points.append(np.array([[x1, y1, z1],
                                [x2, y2, z2],
                                [x3, y3, z3],
                                [x4, y4, z4]]))

    return points

def guess_points_fine_90_2(variables):

    [a, b, c, d] = variables

    points = np.array([[[1.54971192e-02, 0.00000000e+00, 0.00000000e+00],\
                        [-2.62379864e-01, 1.23500882e+01, 0.00000000e+00],\
                        [9.06004094e+00, 2.98725960e+01, 0.00000000e+00],\
                        [2.83695945e+01, 2.98725960e+01, 0.00000000e+00]],\
                       [[0.81549712, 0., 0.],\
                        [0.685709, 17.3050821, 0.],\
                        [12.47546277, 29.74559597, 0.],\
                        [28.36959449, 29.74559597, 0.]],\
                       [[1.01549712, 0., 0.],\
                        [1.01549712, 18.33134059, 0.],\
                        [13.36996099, 29.61859597, 0.],\
                        [28.36959449, 29.61859597, 0.]],\
                       [[1.41549712, 0., 0.],\
                        [2.16869265, 25.10651757, 0.],\
                        [20.05299503, 29.49159597, 0.],\
                        [28.36959449, 29.49159597, 0.]]])

    return points