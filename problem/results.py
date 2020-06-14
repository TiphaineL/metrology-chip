import numpy as np
import matplotlib.pyplot as plt
from curves.Bezier import Bezier_curve

plt.rc('font', size=15)          # controls default text sizes
plt.rc('axes', titlesize=15)     # fontsize of the axes title
plt.rc('axes', labelsize=15)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
plt.rc('legend', fontsize=15)    # legend fontsize
plt.rc('figure', titlesize=15)  # fontsize of the figure title

points_TOF = np.array([[[ 3., 0. ],\
                        [ 3., 28. ],\
                        [50., 28. ]],\
                       [[ 9., 0. ],\
                        [ 9., 27.9],\
                        [50., 27.9]],\
                       [[15., 0. ],\
                        [15., 27.8],\
                        [50., 27.8]],\
                       [[21., 0. ],\
                        [21., 27.7],\
                        [50., 27.7]]])

opd_std_TOF = 5.816371937785698
min_bend_radius_TOF = 20.010640093083243
min_prox_TOF = 0.10121872371455361

Points_fine = np.array([[[ 5.00000000e+01,  2.80000000e+01,  0.00000000e+00],\
                         [ 1.78474700e+01,  2.80000000e+01,  0.00000000e+00],\
                         [ 2.07240347e+01,  2.43866430e+01,  0.00000000e+00],\
                         [ 1.12693752e+01,  2.00084045e+01,  0.00000000e+00],\
                         [ 0.00000000e+00,  1.96700000e+01,  0.00000000e+00]],\

                        [[ 5.00000000e+01,  2.79000000e+01,  0.00000000e+00],\
                         [ 2.21029396e+01,  2.79000000e+01,  0.00000000e+00],\
                         [ 1.90017196e+01,  1.99129034e+01, -7.53001960e-03],\
                         [ 5.81616193e+00,  1.88700000e+01,  0.00000000e+00],\
                         [ 0.00000000e+00,  1.88700000e+01,  0.00000000e+00]],\

                        [[ 5.00000000e+01,  2.78000000e+01,  0.00000000e+00],\
                         [ 2.79979002e+01,  2.78000000e+01,  0.00000000e+00],\
                         [ 1.73343267e+01,  1.83180151e+01,  3.88904867e-01],\
                         [ 7.74649250e+00,  1.86118621e+01,  0.00000000e+00],\
                         [ 0.00000000e+00,  1.86700000e+01,  0.00000000e+00]],\

                        [[ 5.00000000e+01,  2.77000000e+01,  0.00000000e+00],\
                         [ 3.00000000e+01,  2.77000000e+01,  0.00000000e+00],\
                         [ 1.84927346e+01,  1.84495425e+01, -1.09758466e-02],\
                         [ 4.77875687e+00,  1.81623893e+01,  0.00000000e+00],\
                         [ 0.00000000e+00,  1.82700000e+01,  0.00000000e+00]]])

opd_std_fine = 5.588504353524968e-05
min_bend_radius_fine = 61.18487893435561
min_prox_fine = 0.10166126053396404

fig = plt.figure()
for guide in points_TOF:
    curve = Bezier_curve(guide)
    plt.plot(curve.function()[0], curve.function()[1],'r')
for guide in Points_fine:
    curve = Bezier_curve(guide)
    plt.plot(curve.function()[0], curve.function()[1],'r')
plt.show()
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.xlim(0, 50)
plt.ylim(0, 35)

