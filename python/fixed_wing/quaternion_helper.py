"""Port of rl_tools/rl/environments/l2f/quaternion_helper.h

Quaternions are [w, x, y, z] and rotate vectors from the body frame to the world frame.
"""
import numpy as np


def quaternion_multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rotate_vector_by_quaternion(q, v):
    return quaternion_to_rotation_matrix(q) @ v


def rotate_vector_by_quaternion_inverse(q, v):
    return quaternion_to_rotation_matrix(q).T @ v


def quaternion_derivative(q, angular_velocity):
    # angular_velocity is expressed in the body frame
    return 0.5 * quaternion_multiply(q, np.concatenate(([0.0], angular_velocity)))
