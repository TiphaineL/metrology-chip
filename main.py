import numpy as np
import matplotlib.pyplot as plt
from curves.Bezier import Bezier_curve

points = np.array([[ 105., -45.45, 0.],
                   [ 105., -45.45, 9858.32126428],
                   [ 5000.,-700., 20051.12340604],
                   [ 5000., -700., 30000.]])

curve = Bezier_curve(points)

print(curve.points)
print(curve.dimension)
print(curve.order)
print(curve.arc_length())

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.plot(curve.function()[0],curve.function()[1],curve.function()[2])
#eovg2nuw
# ax.set_ylim3d(-2500, 2500)
#ax.set_zlim3d(-2, 2)
ax.grid(False)
# ax.set_aspect('equal')
ax.set_xlabel('X (mm)')
ax.set_ylabel('Z (mm)')
ax.set_zlabel('Y (mm)')
# ax.view_init(-50, 280)
# ax.view_init(-45,270)
# ax.set_axis_off()
plt.show()

fig = plt.figure(2)
bend_radii = 1.0 / curve.curvature()
min_bends = np.min(bend_radii)
plt.plot(bend_radii / 1000)
plt.xlabel('Chip length (arbitrary unit)')
plt.ylabel('Bend radius (mm)')
plt.legend(frameon=False)
plt.ylim([0, 150])
plt.show()
print('min bend', np.min(bend_radii))
