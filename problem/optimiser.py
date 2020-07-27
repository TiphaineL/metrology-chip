from curves.Bezier import Bezier_curve
from problem.cost import cost
from problem.cost_TOFcam import cost_TOF
from problem.solution import solution
import matplotlib.pyplot as plt
import numpy as np
from problem.guess_points import guess_points_fine_straight, guess_points_fine_90deg

#variables = np.array([23,10,28,0,20,9,20,8,0,6,8,20,7,0,6,9,19,8,0,5])
#variables = np.array( [0, 30, 50, np.array([10, 10, 10 ,10]), np.array([25,25,25,25]), np.array([19,19,19,19]), \
#                       np.array([2,2,2,2]), np.array([42,42,42,42]) ])

variables = np.array( [ 2.88170049, 30., 27.10348633,  9.33890891, 14.91898875, 15.66740384, 23.12696949, 15., 15.,\
                        15., 15., 20., 20., 20., 20., 2., 2., 2., 2., 9.02064817, 11.78324052, 12.12532282, 17.33639861])



#variables = np.array([np.random.uniform(0,30),np.random.uniform(0,30),np.random.uniform(0,30),\
#                      np.random.uniform(0,30),np.random.uniform(0,30),np.random.uniform(0,30),np.random.uniform(0,30),\
#                      np.random.uniform(0,30),np.random.uniform(0,30),np.random.uniform(0,30),np.random.uniform(0,30),\
#                      np.random.uniform(0,30),np.random.uniform(0,30),np.random.uniform(0,30),np.random.uniform(0,30),\
#                      np.random.uniform(-3,3),np.random.uniform(-3,3),np.random.uniform(-3,3),np.random.uniform(-3,3),\
#                      np.random.uniform(0,30),np.random.uniform(0,30),np.random.uniform(0,30),np.random.uniform(0,30)])

#variables = np.array([np.random.uniform(0,10),np.random.uniform(20,30),np.random.uniform(20,30),\
#                      np.random.uniform(5,15),np.random.uniform(10,20),np.random.uniform(10,20),np.random.uniform(20,30),\
#                      np.random.uniform(10,20),np.random.uniform(10,20),np.random.uniform(10,20),np.random.uniform(10,20),\
#                      np.random.uniform(15,25),np.random.uniform(15,25),np.random.uniform(15,25),np.random.uniform(15,25),\
#                      np.random.uniform(-3,3),np.random.uniform(-3,3),np.random.uniform(-3,3),np.random.uniform(-3,3),\
#                      np.random.uniform(5,15),np.random.uniform(5,15),np.random.uniform(5,15),np.random.uniform(10,20)])

#variables = np.array([30,50])

test = cost(2)
#test = cost_TOF()

results = test.minimize(variables).x

print('results ',results)
points = guess_points_fine_90deg(results)
#points =test.guess_points(results)

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
