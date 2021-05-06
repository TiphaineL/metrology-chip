from curves.Bezier import Bezier_curve
from problem.cost import cost
from problem.cost_TOFcam import cost_TOF
from problem.solution import solution
import matplotlib.pyplot as plt
import numpy as np
from problem.guess_points import guess_points_TOF_straight, guess_points_fine_90deg

variables_fine_90 = np.array([2.88170049, 30., 27.10348633, 9.33890891, 14.91898875, 15.66740384, 23.12696949, 15., 15., \
                              15., 15., 20., 20., 20., 20., 2., 2., 2., 2., 9.02064817, 11.78324052, 12.12532282, 17.33639861])

variables_TOF_straight = np.array([15,15,15,15,15,15,15,15])

cost_fine_90 = cost(2)
cost_TOF_straight = cost(3)

results = cost_fine_90.minimize(variables_fine_90).x
#results = cost_TOF_straight.minimize(variables_TOF_straight).x
#print('results ',results)
points = guess_points_fine_90deg(results)
#points = guess_points_TOF_straight(results)

#points_TOF_straight = np.array([ [[ 0. , 21. , -0.2 ],[15. , 21. , -0.2 ],[15. , 26.711, -0.2 ],[40. , 26.711, -0.2 ]],\
#                                 [[ 0. , 15. , -0.2 ],[15. , 15. , -0.2 ],[15. , 26.584, -0.2 ],[40. , 26.584, -0.2 ]],\
#                                 [[ 0. , 9. , -0.2 ],[17.82863738, 9. , -0.2 ],[17.01119984, 26.33 , -0.2 ],\
#                                  [40. , 26.33 , -0.2 ]],\
#                                 [[ 0. , 3. , -0.2 ],[22.02534776, 3. , -0.2 ],[17.97291603, 25.949 , -0.2 ],\
#                                  [40. , 25.949 , -0.2 ]]])

#points_fine_90 = np.array([[[ 2.71048907, 0. , -0.1 ],[ 2.4346672 , 12.25875001, -0.1 ],[11.51278496, 17.07540903, -0.17 ],\
#                            [11.59413006, 29.873 , -0.1 ],[31.71868039, 29.873 , -0.1 ]],\
#                           [[ 3.51048907, 0. , -0.1 ],[ 3.37907236, 17.52222875, -0.1 ],[11.53723978, 19.82565246, -0.03 ],\
#                            [14.76440076, 29.746 , -0.1 ],[31.71868039, 29.746 , -0.1 ]],\
#                           [[ 3.71048907, 0. , -0.1 ],[ 3.71048907, 17.94273665, -0.1 ],[11.41824356, 19.86042721, -0.03 ],\
#                            [15.05847058, 29.619 , -0.1 ],[31.71868039, 29.619 , -0.1 ]],\
#                           [[ 4.11048907, 0. , -0.1 ],[ 4.83791766, 24.24761969, -0.1 ],[15.85286502, 19.95488547, -0.03 ],\
#                            [14.04254114, 29.492 , -0.1 ],[31.71868039, 29.492 , -0.1 ]]])

#points = points_TOF_straight
#points = points_fine_90

curve_1 = Bezier_curve(points[0])
curve_2 = Bezier_curve(points[1])
curve_3 = Bezier_curve(points[2])
curve_4 = Bezier_curve(points[3])

solution_ = solution(curve_1,curve_2,curve_3,curve_4)

solution_.plot_waveguides()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(curve_1.function()[0], curve_1.function()[1], curve_1.function()[2])
ax.plot(curve_2.function()[0], curve_2.function()[1], curve_2.function()[2])
ax.plot(curve_3.function()[0], curve_3.function()[1], curve_3.function()[2])
ax.plot(curve_4.function()[0], curve_4.function()[1], curve_4.function()[2])
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')

solution_.plot_distances()
solution_.plot_bend_radii()
solution_.print_arclength()
print('Points ',points)
print('OPD std ',solution_.standDev_OPD())

print('bend radius ',solution_.min_bend_radius())
print('proximity ',solution_.min_proximity())
