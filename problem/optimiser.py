from curves.Bezier import Bezier_curve
from problem_variables.initial_conditions import *
from problem.cost import cost
from problem.solution import solution

#def guess_points(IC, w, angles):
    # x = IC[4]
    # point_1 = np.array([[10, IC[0]], \
    #                     [x - 7 * w + IC[0] * np.tan(angles[0] * np.pi / 180), IC[0]], \
    #                     [x - 7 * w, 0]])
    #
    # point_2 = np.array([[10, IC[1]], \
    #                     [x - 3 * w + IC[1] * np.tan(angles[1] * np.pi / 180), IC[1]],
    #                     [x - 3 * w, 0]])
    #
    # point_3 = np.array([[10, IC[2]], \
    #                     [x - 2 * w + IC[2] * np.tan(angles[2] * np.pi / 180), IC[2]], \
    #                     [x - 2 * w, 0]])
    #
    # point_4 = np.array([[10, IC[3]], \
    #                     [x + IC[3] * np.tan(angles[3] * np.pi / 180), IC[3]], \
    #                     [x, 0]])
    #
    # return np.array([point_1, point_2, point_3, point_4])

variables = np.array([h1, h2, h3, h4, x])

test = cost()
results = test.minimize(variables).x

print('variables ',variables)
print('results ',results)
points = test.guess_points(results)
#points = guess_points(variables,x,w,angles)
curve_1 = Bezier_curve(points[0])
curve_2 = Bezier_curve(points[1])
curve_3 = Bezier_curve(points[2])
curve_4 = Bezier_curve(points[3])

solution_ = solution(curve_1,curve_2,curve_3,curve_4)
solution_.plot_waveguides()
solution_.plot_distances()
solution_.plot_bend_radii()
solution_.print_arclength()
print('OPD std ',solution_.standDev_OPD())
print('bend radius ',solution_.min_bend_radius())
print('proximity ',solution_.min_proximity())
print('cost ',test.function(variables))

#cost_F = cost()#_,[20,100],[.3,.7],[1e-3,.1e-3])
#sol = cost_F.minimize()
#print(sol)

#def cost_test(arg):
#    return 1+ 2*arg[0] - arg[1]

#cost_F = cost(solution_,[20,100],[.3,.7],[1e-3,.1e-3])
#cost_F = cost_F.function()


#optimize.minimize(cost(solution_,[20,100],[.3,.7],[1e-3,.1e-3]).minimize, variables, method='Nelder-Mead', tol=1e-6)


#sol = scipy.optimize.minimize(cost_F,\
#                              variables,\
#                                method='Powell',bounds= bounds)