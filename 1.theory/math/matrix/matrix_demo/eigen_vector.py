import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from math import sqrt
from math import pi

def line(v):
    if abs(v[0]) > abs(v[1]):
        x = l+1
        y = x/v[0]*v[1]
    else:
        y = l+1
        x = y/v[1]*v[0]
    return [x, y]


a = np.array([[2,1],[2,3]])
a = np.array([[3,3],[2,1]])
a = np.array([[-2,1],[-1,0]])
a = np.array([[2,1],[-1,0]])
a = np.array([[0,-1],[1,0]])
a = np.array([[2,-1],[3,0]])
# upper and lower limit
l = 3

x,y = np.meshgrid(np.arange(-l-0.1, l+0.1, 0.5), np.arange(-l-0.1, l+0.1, 0.5))

# difference
u = np.array(np.zeros([len(x),len(x[0])]))
v = np.array(np.zeros([len(x),len(x[0])]))
# ratio
r = np.array(np.zeros([len(x),len(x[0])]))
for i in range(len(x)):
    for j in range(len(x[0])):
        vec = np.array([x[i,j],y[i,j]])
        vec_p = np.dot(a,vec)
        u[i, j] = (vec_p[0] - vec[0])
        v[i, j] = (vec_p[1] - vec[1])

fig = plt.figure('eigen_vector')
ax1 = fig.add_subplot(1, 3, 1)
ax1.set_title('difference')
ax1.set_aspect(1)
ax1.quiver(x,y,u,v)
# ax1.quiver(x,y,u,v, units='xy', angles='xy', scale_units='xy', scale=1.)
eigen_val, eigen_vec = np.linalg.eig(a)
print(eigen_val)
# eigen values and eigen vectors are real numbers/vectors
if np.real(eigen_val[0]) == eigen_val[0]:
    line1 = line(eigen_vec[:,0].T)
    ax1.plot([-line1[0], line1[0]], [-line1[1], line1[1]], linestyle='--', c='r', lw=3, alpha=0.5)

    line2 = line(eigen_vec[:,1].T)
    ax1.plot([-line2[0], line2[0]], [-line2[1], line2[1]], linestyle='-.', c='g', lw=3, alpha=0.5)

plt.xlim([-l,l])
plt.ylim([-l,l])
plt.grid()

x,y = np.meshgrid(np.arange(-3, 3, 0.1), np.arange(-3, 3, 0.1))
# ratio
r = np.array(np.zeros([len(x),len(x[0])]))

for i in range(len(x)):
    for j in range(len(x[0])):
        vec = np.array([x[i,j],y[i,j]])
        vec_p = np.dot(a,vec)
        d = sqrt(vec[0] ** 2 + vec[1] ** 2)
        d_p = sqrt(vec_p[0] ** 2 + vec_p[1] ** 2)
        if d == 0:
            r[i,j] = 0
        else:
            r[i,j] = d_p / d
            # might change the direction?

ax2 = fig.add_subplot(1, 3, 2, projection='3d')
# ax2.set_aspect(1)
# ax2.set_aspect('equal')
ax2.set_title('ratio for mod')

# ax2.plot_surface(x,y,r)
ax2.plot_wireframe(x,y,r, alpha=0.3)
if np.real(eigen_val[0]) == eigen_val[0]:
    x1 = np.linspace(-line1[0], line1[0], 10)
    x2 = np.linspace(-line2[0], line2[0], 10)
    y1 = np.linspace(-line1[1], line1[1], 10)
    y2 = np.linspace(-line2[1], line2[1], 10)
    r1 = eigen_val[0] if eigen_val[0] > 0 else -eigen_val[0]
    r2 = eigen_val[1] if eigen_val[1] > 0 else -eigen_val[1]
    # z1 = np.array([eigen_val[0]]*len(x1))
    # z2 = np.array([eigen_val[1]]*len(x1))
    z1 = np.array([r1]*len(x1))
    z2 = np.array([r2]*len(x1))

    ax2.plot(x1,y1,z1, c='r', linestyle='--')
    ax2.plot(x2,y2,z2, c='g', linestyle='-.')
ax2.set_xlim([-3,3])
ax2.set_ylim([-3,3])
# plt.grid()
ax2.set_zlim(0.5, 4)

ax3 = fig.add_subplot(1, 3, 3, projection='3d')
# ax2.set_aspect(1)
# ax2.set_aspect('equal')
ax3.set_title('ratio with direction')

for i in range(len(x)):
    for j in range(len(x[0])):
        vec = np.array([x[i,j],y[i,j]])
        vec_p = np.dot(a,vec)
        d = sqrt(vec[0] ** 2 + vec[1] ** 2)
        d_p = sqrt(vec_p[0] ** 2 + vec_p[1] ** 2)
        if d == 0:
            r[i,j] = 0
        else:
            r[i,j] = d_p / d
            # might change the direction?
            cos_vec = vec_p.dot(vec)/(d*d_p)
            if cos_vec > 1:
                cos_vec = 1.0-10**-6
            if cos_vec < -1:
                cos_vec = -1.0+10**-6
            angle = np.arccos(cos_vec)
            if angle >= pi/2:
                r[i, j] = -d_p/d
            if np.isnan(angle):
                print(vec_p.dot(vec)/(d*d_p), end=', ')
                print(vec_p.dot(vec), end=', ')
                print(d, end=', ')
                print(d_p, end=', ')
                print(angle)

ax3.plot_wireframe(x,y,r, alpha=0.3)
if np.real(eigen_val[0]) == eigen_val[0]:
    x1 = np.linspace(-line1[0], line1[0], 10)
    x2 = np.linspace(-line2[0], line2[0], 10)
    y1 = np.linspace(-line1[1], line1[1], 10)
    y2 = np.linspace(-line2[1], line2[1], 10)
    z1 = np.array([eigen_val[0]]*len(x1))
    z2 = np.array([eigen_val[1]]*len(x1))

    ax3.plot(x1,y1,z1, c='r', linestyle='--')
    ax3.plot(x2,y2,z2, c='g', linestyle='-.')
ax3.set_xlim([-3,3])
ax3.set_ylim([-3,3])
# plt.grid()
# ax3.set_zlim(0.5, 4)

plt.show()
