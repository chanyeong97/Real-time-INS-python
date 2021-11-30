import numpy as np


def angular_velocity_to_quaternion_derivative(q, w):
    omega = np.array([[0, -w[0], -w[1], -w[2]],
                      [w[0], 0, w[2], -w[1]],
                      [w[1], -w[2], 0, w[0]],
                      [w[2], w[1], -w[0], 0]]) * 0.5
    return np.dot(omega, q)


def gyro_integration(ts, gyro, init_q):
    output_q = np.zeros((gyro.shape[0], 4))
    output_q[0] = init_q
    dts = ts[1:] - ts[:-1]
    for i in range(1, gyro.shape[0]):
        output_q[i] = output_q[i - 1] + angular_velocity_to_quaternion_derivative(output_q[i - 1], gyro[i - 1]) * dts[
            i - 1]
        output_q[i] /= np.linalg.norm(output_q[i])
    return output_q