from curves.Bezier import Bezier_curve
from problem_variables.initial_conditions import *
from problem.cost import cost
from problem.solution import solution
import matplotlib.pyplot as plt

variables = np.array([h1, h2, h3, h4, x])
variables = np.array([0,10])
variables = np.array([23,10,28,0,20,9,20,8,0,6,8,20,7,0,6,9,19,8,0,5])

def guess_points(w,angles):

    x0 = 30
    y0 = 10

    x = 10

    point_1 = np.array([[x0, y0], \
                        [x+7 * w + y0 * np.tan(angles[0] * np.pi / 180), y0], \
                        [x+7 * w, 0]])

    point_2 = np.array([[x0, y0 - 0.01], \
                        [x+3 * w + (y0 - 0.01) * np.tan(angles[1] * np.pi / 180), y0-0.01],
                        [x+3 * w, 0]])

    point_3 = np.array([[x0, y0 - 0.02], \
                        [x+2 * w + (y0 - 0.02) * np.tan(angles[2] * np.pi / 180), y0-0.02], \
                        [x+2 * w, 0]])

    point_4 = np.array([[x0, y0 - 0.03], \
                        [x+(y0-0.03)* np.tan(angles[3] * np.pi / 180), y0-0.03], \
                        [x+0, 0]])

    return np.array([point_1, point_2, point_3, point_4])

test = cost()
results = test.minimize(variables).x

print('results ',results)
points = test.guess_points(results)
#points = guess_points(variables,w,angles)

curve_1 = Bezier_curve(points[0])
curve_2 = Bezier_curve(points[1])
curve_3 = Bezier_curve(points[2])
curve_4 = Bezier_curve(points[3])

solution_ = solution(curve_1,curve_2,curve_3,curve_4)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(curve_1.function()[0], curve_1.function()[1], curve_1.function()[2])
ax.plot(curve_2.function()[0], curve_2.function()[1], curve_2.function()[2])
ax.plot(curve_3.function()[0], curve_3.function()[1], curve_3.function()[2])
ax.plot(curve_4.function()[0], curve_4.function()[1], curve_4.function()[2])

solution_.plot_distances()
solution_.plot_bend_radii()
solution_.print_arclength()
print('Points ',points)
print('OPD std ',solution_.standDev_OPD())

print('bend radius ',solution_.min_bend_radius())
print('proximity ',solution_.min_proximity())
