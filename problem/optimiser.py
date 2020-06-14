from curves.Bezier import Bezier_curve
from problem.cost import cost
from problem.cost_TOFcam import cost_TOF
from problem.solution import solution
import matplotlib.pyplot as plt

variables = np.array([23,10,28,0,20,9,20,8,0,6,8,20,7,0,6,9,19,8,0,5])
#variables = np.array([30, 25])

test = cost()
#test = cost_TOF()

results = test.minimize(variables).x

print('results ',results)
points = test.guess_points(results)

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
#ax.set_xlabel('X (mm)')
#ax.set_ylabel('Z (mm)')
#ax.set_zlabel('Y (mm)')

solution_.plot_distances()
solution_.plot_bend_radii()
solution_.print_arclength()
print('Points ',points)
print('OPD std ',solution_.standDev_OPD())

print('bend radius ',solution_.min_bend_radius())
print('proximity ',solution_.min_proximity())
