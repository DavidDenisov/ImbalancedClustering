from numpy.random import default_rng
import numpy as np
import matplotlib.pyplot as plt
rng = default_rng()

# Nuber of points in each cluster:
n1,n2 = 1250,25

# Centers:
c1,c2 = [0,0],[2.25, 0]

# Radius:
r1,r2 = 1,0.1

# Helper function te generate num points in a circle, with center in c, and radius r.
def sample_circle(c: [float,float],r : float,num: int):
    # Random angle
    alpha = 2 * np.pi * rng.random(num)
    # Random radius
    r = r * np.sqrt(rng.random(num))
    # Calculating coordinates
    x = r * np.cos(alpha) + c[0]
    y = r * np.sin(alpha) + c[1]
    P = np.concatenate(([x], [y]), 0).T
    return P

# Set plot parameters
plt.rcParams['figure.figsize'] = (16, 9)
plt.rcParams.update({'font.size': 50})
plt.xlabel("x")
plt.ylabel("y")

# Sample the points
A = sample_circle(c1,r1,n1)
B = sample_circle(c2,r2, n2)

# Add the points to the plot
plt.scatter(A[:, 0], A[:, 1], c='r', s=250)
plt.scatter(B[:, 0], B[:, 1], c='b', s=250)

# Show the plot
plt.show()