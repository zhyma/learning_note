import numpy as np
import matplotlib.pyplot as plt

a = np.array([[2,1],[2,3]])
v_x0 = np.array([[1],[0]])
v_y0 = np.array([[0],[1]])
v_x1 = np.array([[3/4],[-1/2]])
v_y1 = np.array([[-1/4],[1/2]])
v_1 = np.array([[1],[-1]])
v_2 = np.array([[1],[2]])

def f(t):
    return np.exp(-t) * np.cos(2 * np.pi * t)
 
 
t1 = np.arange(0, 5, 0.1)
t2 = np.arange(0, 5, 0.02)

fig = plt.figure('eigen')
ax1 = fig.add_subplot(121)
ax1.arrow(0, 0, 1, 0, length_includes_head=True, head_width=0.1, lw=2, head_length=0.2, color='g')
ax1.text(1, 0, "x", ha="center", va="center", size=20)

ax1.arrow(0, 0, 0, 1, length_includes_head=True, head_width=0.1, lw=2, head_length=0.2, color='g')
ax1.text(0, 1, "y", ha="center", va="center", size=20)

ax1.arrow(0, 0, 1, -1, length_includes_head=True, head_width=0.1, lw=2, head_length=0.2, color='b')
ax1.text(1, -1, "v1", ha="center", va="center", size=20)

ax1.arrow(0, 0, 1, 2, length_includes_head=True, head_width=0.1, lw=2, head_length=0.2, color='b')
ax1.text(1, 2, "v2", ha="center", va="center", size=20)

ax1.arrow(0, 0, 3/4.0, -1/2.0, length_includes_head=True, head_width=0.1, lw=2, head_length=0.2, color='r')
ax1.text(3/4.0, -1/2.0, "x'", ha="center", va="center", size=20)

ax1.arrow(0, 0, -1/4.0, 1/2.0, length_includes_head=True, head_width=0.1, lw=2, head_length=0.2, color='r')
ax1.text(-1/4.0, 1/2.0, "y'", ha="center", va="center", size=20)

plt.xlim([-4,4])
plt.ylim([-4,4])
plt.grid()
ax1.set_aspect(1)
# plt.plot(t1, f(t1), 'bo', t2, f(t2), 'r--')
 
ax2 = fig.add_subplot(122)

ax2.arrow(0, 0, 1, 0, length_includes_head=True, head_width=0.1, lw=2, head_length=0.2, color='r')
ax2.text(1, 0, "x'", ha="center", va="center", size=20)

ax2.arrow(0, 0, 0, 1, length_includes_head=True, head_width=0.1, lw=2, head_length=0.2, color='r')
ax2.text(0, 1, "y'", ha="center", va="center", size=20)

ax2.arrow(0, 0, 1, -1, length_includes_head=True, head_width=0.1, lw=2, head_length=0.2, color='b')
ax2.text(1, -1, "v1", ha="center", va="center", size=20)

ax2.arrow(0, 0, 1, 2, length_includes_head=True, head_width=0.1, lw=2, head_length=0.2, color='b')
ax2.text(1, 2, "v2", ha="center", va="center", size=20)

ax2.arrow(0, 0, 2, 2, length_includes_head=True, head_width=0.1, lw=2, head_length=0.2, color='g')
ax2.text(2, 2, "x'", ha="center", va="center", size=20)

ax2.arrow(0, 0, 1, 3, length_includes_head=True, head_width=0.1, lw=2, head_length=0.2, color='g')
ax2.text(1, 3, "y'", ha="center", va="center", size=20)

plt.xlim([-4,4])
plt.ylim([-4,4])
plt.grid()
ax2.set_aspect(1)
# plt.plot(t2, np.cos(2 * np.pi * t2), 'r--')
 
 
plt.show()