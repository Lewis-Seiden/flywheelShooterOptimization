#!/usr/bin/env python3

"""FRC 2022 shooter trajectory optimization (CasADi + Ipopt).

This program finds the problemmal initial launch velocity and launch angle for the
2024 FRC game's target.

It is derived from a model made by Tyler.
"""

import math

import casadi as ca
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.linalg import norm

field_width = 8.2296  # 27 ft
field_length = 16.4592  # 54 ft

# Robot initial velocity
robot_initial_v_x = 0.0  # ft/s
robot_initial_v_y = 0.0  # ft/s
robot_initial_v_z = 0.0  # ft/s

g = 9.806
max_launch_velocity = 25.0

shooter = np.array([[field_length / 3.0], [field_width / 3.0], [1.2]])
print(shooter)
shooter_x = shooter[0, 0]
shooter_y = shooter[1, 0]
shooter_z = shooter[2, 0]

target = np.array([[0.0], [5.476], [2.2]])
print(target)
target_x = target[0, 0]
target_y = target[1, 0]
target_z = target[2, 0]
target_radius = 0.61


def lerp(a, b, t):
    return a + t * (b - a)


problem = ca.Opti()

# Set up duration decision variables
N = 50
T = problem.variable()
problem.subject_to(T >= 0)
problem.set_initial(T, 1)
dt = T / N

#     [x position]
#     [y position]
#     [z position]
# x = [x velocity]
#     [y velocity]
#     [z velocity]
state = problem.variable(6, N)

p_x = state[0, :]
p_y = state[1, :]
p_z = state[2, :]
v_x = state[3, :] + np.full((1, N), robot_initial_v_x)
v_y = state[4, :] + np.full((1, N), robot_initial_v_y)
v_z = state[5, :] + np.full((1, N), robot_initial_v_z)

# Position initial guess is linear interpolation between start and end position
for k in range(N):
    problem.set_initial(p_x[k], lerp(shooter_x, target_x, k / N))
    problem.set_initial(p_y[k], lerp(shooter_y, target_y, k / N))
    problem.set_initial(p_z[k], lerp(shooter_z, target_z, k / N))

# Velocity initial guess is max launch velocity toward goal
uvec_shooter_to_target = target - shooter
uvec_shooter_to_target /= norm(uvec_shooter_to_target)
for k in range(N):
    problem.set_initial(state[3, k], max_launch_velocity * uvec_shooter_to_target[0, 0])
    problem.set_initial(state[4, k], max_launch_velocity * uvec_shooter_to_target[1, 0])
    problem.set_initial(state[5, k], max_launch_velocity * uvec_shooter_to_target[2, 0])

# Shooter initial position
problem.subject_to(state[:3, 0] == shooter)

# Require initial launch velocity is below max
# √{v_x² + v_y² + v_z²) <= vₘₐₓ
# v_x² + v_y² + v_z² <= vₘₐₓ²
problem.subject_to(v_x[0] ** 2 + v_y[0] ** 2 + v_z[0] ** 2 <= max_launch_velocity**2)

# Require final position is in center of target circle
problem.subject_to(p_x[-1] == target_x)
problem.subject_to(p_y[-1] == target_y)
problem.subject_to(p_z[-1] == target_z)

def calc_forces(x):
    # x' = x'
    # y' = y'
    # z' = z'
    # x" = −a_D(v_x)
    # y" = −a_D(v_y)
    # z" = −g − a_D(v_z)
    #
    # where a_D(v) = ½ρv² C_D A / m
    rho = 1.204  # kg/m³
    m = 0.24  # kg
    # small side
    C_Ds = 0.5
    A_s = 0.018
    # face-on
    C_Df = 0.5
    A_f = 0.0486
    
    a_D = lambda v, c_d, a: 0.5 * rho * v**2 * c_d * a / m

    v_x = x[3, 0]
    v_y = x[4, 0]
    v_z = x[5, 0]
    return ca.vertcat(v_x, v_y, v_z, -a_D(v_x, C_Ds, A_s), -a_D(v_y, C_Ds, A_s), -g - a_D(v_z, C_Df, A_f))


# Dynamics constraints - RK4 integration
for k in range(N - 1):
    h = dt
    x_k = state[:, k]
    x_k1 = state[:, k + 1]

    k1 = calc_forces(x_k)
    k2 = calc_forces(x_k + h / 2 * k1)
    k3 = calc_forces(x_k + h / 2 * k2)
    k4 = calc_forces(x_k + h * k3)
    problem.subject_to(x_k1 == x_k + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4))

# Minimize distance from goal over time
J = 0
for k in range(N):
    J += (target - state[:3, k]).T @ (target - state[:3, k])
problem.minimize(J)

problem.solver("ipopt")
sol = problem.solve()

# Initial velocity vector
v = sol.value(state[3:, 0])

launch_velocity = norm(v)
print(f"Launch velocity = {round(launch_velocity, 3)} m/s")

# The launch angle is the angle between the initial velocity vector and the x-y
# plane. First, we'll find the angle between the z-axis and the initial velocity
# vector.
#
# sinθ = |a x b| / (|a| |b|)
#
# Let v be the initial velocity vector and p be a unit vector along the z-axis.
#
# sinθ = |v x p| / (|v| |p|)
# sinθ = |v x [0, 0, 1]| / |v|
# sinθ = |[v_y, -v_x, 0]|/ |v|
# sinθ = √(v_x² + v_y²) / |v|
#
# The square root part is just the norm of the first two components of v.
#
# sinθ = |v[:2]| / |v|
#
# θ = asin(|v[:2]| / |v|)
#
# The angle between the initial velocity vector and the X-Y plane is 90° − θ.
launch_angle = math.pi / 2.0 - math.asin(norm(v[:2]) / norm(v))
print(f"Launch angle = {round(launch_angle * 180.0 / math.pi, 3)}°")

fig = plt.figure()
ax = plt.axes(projection="3d")


def plot_wireframe(ax, f, x_range, y_range, color):
    x, y = np.mgrid[x_range[0] : x_range[1] : 25j, y_range[0] : y_range[1] : 25j]

    # Need an (N, 2) array of (x, y) pairs.
    xy = np.column_stack([x.flat, y.flat])

    z = np.zeros(xy.shape[0])
    for i, pair in enumerate(xy):
        z[i] = f(pair[0], pair[1])
    z = z.reshape(x.shape)

    ax.plot_wireframe(x, y, z, color=color)


# Ground
plot_wireframe(ax, lambda x, y: 0.0, [0, field_length], [0, field_width], "grey")

# Target
ax.plot(
    target_x,
    target_y,
    target_z,
    color="black",
    marker="x",
)
xs = []
ys = []
zs = []
for angle in np.arange(0.0, 2.0 * math.pi, 0.1):
    xs.append(target_x)
    ys.append(target_y + target_radius * math.cos(angle))
    zs.append(target_z + target_radius * math.sin(angle))
ax.plot(xs, ys, zs, color="black")

# Trajectory
trajectory_x = sol.value(p_x)
trajectory_y = sol.value(p_y)
trajectory_z = sol.value(p_z)
ax.plot(trajectory_x, trajectory_y, trajectory_z, color="orange")

ax.set_box_aspect((field_length, field_width, np.max(trajectory_z)))

ax.set_xlabel("X position (m)")
ax.set_ylabel("Y position (m)")
ax.set_zlabel("Z position (m)")

plt.show()
