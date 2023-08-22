import numpy as np
import matplotlib.pyplot as plt
from tudatpy.kernel.astro import element_conversion

# Test
data = np.load("..\\Data\\Temp\\control5.npz")
# data = np.load("..\\Data\\Temp\\no_control.npz")
oe = data['oe_sat']
oe_ref = data['oe_ref']
thrust_forces = data['inputs']

satellite = 0

a = oe[:-1, 0]
e = oe[:-1, 1]
i = oe[:-1, 2]
periapsis = np.unwrap(oe[:-1, 3], axis=0)
RAAN = np.unwrap(oe[:-1, 4], axis=0)
true_anomaly = np.unwrap(oe[:-1, 5], axis=0)
arg_of_lat = true_anomaly + periapsis

RAAN_ref = np.unwrap(oe_ref[0, :-1, 4], axis=0)
inc_ref = oe_ref[0, :-1, 2]

true_anomaly_at_start = true_anomaly[(np.arange(0, true_anomaly.shape[0]) // 20) * 20]

kappa = 1 + e * np.cos(true_anomaly)
eta = 1 - e**2
r = a * eta / kappa

F_r = thrust_forces[1:, 0, 0]
F_theta = thrust_forces[1:, 1, 0]
F_z = thrust_forces[1:, 2, 0]

mean_motion = np.sqrt(100 / 55**3)
satellite_mass = 400
a_dot = 2 * F_theta / mean_motion / satellite_mass
e_dot = (np.sin(true_anomaly) * F_r + 2 * np.cos(true_anomaly) * F_theta) / mean_motion / satellite_mass / 55

e_dot_at_start = (np.sin(true_anomaly_at_start) * F_r + 2 * np.cos(true_anomaly_at_start) * F_theta) / mean_motion / satellite_mass / 55

eta = 1 - e**2
kappa = 1 + e * np.cos(true_anomaly)

actual_mean_motion = np.sqrt(100 / a**3)
actual_mean_motion_2 = np.sqrt(100 / r**3)

e_true_anomaly_dot = ((eta * (kappa * np.cos(true_anomaly) - 2 * e)) / kappa * F_r - (eta * (1 + kappa) * np.sin(true_anomaly)) / kappa * F_theta) / mean_motion / satellite_mass / 55 + mean_motion * e
e_true_dot_simple = (np.cos(true_anomaly) * F_r - 2 * np.sin(true_anomaly) * F_theta) / mean_motion / satellite_mass / 55 + mean_motion * e + 2 * e_dot * e * np.sin(true_anomaly)
true_anomaly_dot = np.gradient(true_anomaly, axis=0, edge_order=2)
periapsis_dot = np.gradient(periapsis, axis=0, edge_order=2)
true_latitude_dot = np.gradient(np.unwrap(periapsis + true_anomaly, axis=0), axis=0, edge_order=2)

t = np.arange(0, a.shape[0]) / 60

# plt.figure()
# plt.plot(t, np.gradient(a, axis=0, edge_order=2))
# plt.plot(t, a_dot)
# plt.figure()
# plt.title('e dot')
# plt.plot(t, np.gradient(e, axis=0, edge_order=2))
# plt.plot(t, e_dot)
# plt.plot(t, e_dot_at_start)
# plt.figure()
# plt.title("Eccentricity")
# plt.plot(t, e)
# plt.figure()
# plt.title("Semi-major axis")
# plt.plot(t, a)
# plt.figure()
# plt.title("kappa and eta")
# plt.plot(t, kappa, label='kappa')
# plt.plot(t, eta, label='eta')
# plt.legend()
#
# plt.figure()
# plt.title("R dot")
# plt.plot(t, np.gradient(e, axis=0, edge_order=2))
# plt.plot(t, e_dot)

# plt.figure()
# plt.title("e f dot")
# plt.plot(t, e * true_anomaly_dot)
# plt.plot(t, e_true_anomaly_dot)
# plt.plot(t, e_true_dot_simple, '--')
#
# plt.figure()
# plt.title("Raw angles")
# plt.plot(t, true_anomaly + periapsis, label='true latitude')
# plt.plot(t, periapsis, label='periapsis')
# plt.plot(t, true_anomaly, label='true_anomaly')
# # plt.plot(t, true_anomaly_at_start, label='start_anomaly')
# plt.legend()

# plt.figure()
# plt.title("True lat dot")
# plt.plot(t[2:], true_latitude_dot[2:] - mean_motion, label='true')
# # plt.plot(t, (-2 * F_r - np.sin(true_anomaly + periapsis) / np.tan(i) * F_z) / actual_mean_motion_2 / a / satellite_mass, label='roe')
# # plt.plot(t, np.cumsum(F_theta / r / satellite_mass), label='hcw')
# # plt.plot(t, - np.sin(true_anomaly + periapsis) / np.tan(i) * F_z / actual_mean_motion_2 / r / satellite_mass, label='Fz')
# plt.plot(t, e * np.cos(true_anomaly) * actual_mean_motion_2 / 2 - np.sin(true_anomaly + periapsis) / np.tan(i) * F_z / actual_mean_motion_2 / r / satellite_mass, '-.', label='test')
# plt.plot(t, - 3/2 * mean_motion * e * np.cos(true_anomaly) - 3/2 * e * np.sin(true_anomaly) * mean_motion, '-.', label='Add')
# # plt.plot(t, , '-.', label='Add2')
# plt.plot(t, e * np.cos(true_anomaly) * actual_mean_motion_2 / 2 - np.sin(true_anomaly + periapsis) / np.tan(i) * F_z / actual_mean_motion_2 / r / satellite_mass-3/2 * mean_motion * (r - 55) / 55 - 0 * 3/2 * mean_motion * e * np.cos(true_anomaly) + 0 * 3/2 * e * np.cos(true_anomaly) * mean_motion, '-.', label='test')
# plt.legend()
# # plt.plot(t, actual_mean_motion)
# # plt.plot(t, actual_mean_motion_2)

# plt.figure()
# plt.title("Inputs")
# plt.plot(t, F_r, label='F_r')
# plt.plot(t, F_theta, label='F_theta')
# plt.plot(t, F_z, label='F_z')
# plt.legend()

# plt.figure()
# plt.title("Limit of e for latitude")
# e_test = np.logspace(-9, -2, num=1000)
# f_test = np.deg2rad(30)
#
# eta_test = np.sqrt(1 - e_test**2)
# kappa_test = 1 + e_test * np.cos(f_test)
#
# formula_1 = -((eta_test**2 - eta_test) * (1 + kappa_test) * np.cos(f_test) - 2 * e_test * eta_test**2) / e_test / kappa_test
# formula_2 = ((eta_test - eta_test**2) * (1 + kappa_test) * np.sin(f_test)) / e_test / kappa_test
# plt.loglog(e_test, formula_1, label='f1')
# plt.loglog(e_test, formula_2, label="f2")
# plt.legend()

# plt.figure()
# plt.title("Theta dot")
# # M = np.zeros_like(true_anomaly)
# # for i in range(t.shape[0]):
# #     M[i] = element_conversion.true_to_mean_anomaly(e[i], true_anomaly[i])
#
# plt.plot(t, true_anomaly_dot)
# # plt.plot(t, np.gradient(M, axis=0, edge_order=2), '--')
# plt.plot(t[1:], mean_motion + (2 * np.cos(true_anomaly) * F_r - 2 * np.sin(true_anomaly) * F_theta)[1:] / e[1:] / mean_motion / 55 / satellite_mass)
# plt.plot(t[1:], actual_mean_motion[1:] + (2 * np.cos(true_anomaly) * F_r - 2 * np.sin(true_anomaly) * F_theta)[1:] / e[1:] / actual_mean_motion[1:] / 55 / satellite_mass, '--')
# plt.plot(t[1:], actual_mean_motion_2[1:] + (2 * np.cos(true_anomaly) * F_r - 2 * np.sin(true_anomaly) * F_theta)[1:] / e[1:] / actual_mean_motion_2[1:] / 55 / satellite_mass, '-.')
# plt.plot(t[1:], actual_mean_motion[1:])
# plt.plot(t[1:], actual_mean_motion_2[1:])

# plt.figure()
# plt.title("R dot")
# plt.plot(t, np.gradient(r, axis=0, edge_order=2))
# last_term = 2 * e * e_dot * a / kappa
#
# r_dot_theory = a_dot * eta / kappa - 2 * e * e_dot * a / kappa - np.cos(true_anomaly) * e_dot * a * eta / kappa**2 + true_anomaly_dot * np.sin(true_anomaly) * e * a * eta / kappa**2
# # plt.plot(t, a_dot - e_dot * a * np.cos(true_anomaly) + orbital_sim.mean_motion * np.sin(true_anomaly) * e * a)
# # plt.plot(t, a_dot - e_dot * a * np.cos(true_anomaly) + true_anomaly_dot * np.sin(true_anomaly) * e * a, '--')
# plt.plot(t, a_dot - e_dot * 55 * np.cos(true_anomaly) + e_true_dot_simple * np.sin(true_anomaly) * 55, '.-')
# plt.plot(t, actual_mean_motion_2 * e * np.sin(true_anomaly) * 55)
#
# plt.figure()
# plt.title("R dot dot")
# plt.plot(t, np.gradient(np.gradient(r, axis=0, edge_order=2), axis=0, edge_order=2))
# plt.plot(t, F_r / satellite_mass + mean_motion**2 * e * a * np.cos(true_anomaly))
# plt.plot(t, true_anomaly_dot * np.cos(true_anomaly) * mean_motion * e * a + e_dot * mean_motion * a * np.sin(true_anomaly))
# # plt.plot(t, r_dot_theory, '--')
# # plt.plot(t, a_dot - e_dot * a * np.cos(true_anomaly) + e_true_anomaly_dot * np.sin(true_anomaly) * a - last_term, '--')
# # plt.plot(t, last_term)
# # plt.plot(t, true_anomaly_dot * np.sin(true_anomaly) * e * a - 2 * e * e_dot * a)
#
# plt.figure()
# plt.title("(e cos f) dot")
# plt.plot(t, np.gradient(e * np.cos(true_anomaly), axis=0, edge_order=2))
# plt.plot(t, 2 * F_theta / satellite_mass / a / actual_mean_motion_2 - actual_mean_motion_2 * e * np.sin(true_anomaly))
# #
# plt.figure()
# plt.title("(e sin f) dot")
# plt.plot(t, np.gradient(e * np.sin(true_anomaly), axis=0, edge_order=2))
# plt.plot(t, F_r / a / actual_mean_motion_2 / satellite_mass + actual_mean_motion_2 * e * np.cos(true_anomaly))

# plt.figure()
# i_dot = np.gradient(i, axis=0, edge_order=2)
# RAAN_dot = np.gradient(RAAN, axis=0, edge_order=2)
# plt.plot(t, np.sqrt(i_dot**2 + np.sin(i)**2 * RAAN_dot**2))
# plt.plot(t, np.abs(F_z) / a / actual_mean_motion_2 / satellite_mass)

# plt.figure()
# cos_theta_sin_i = np.cos(arg_of_lat) * np.sin(i)
# sin_theta_sin_i = np.sin(arg_of_lat) * np.sin(i)
# plt.plot(t, mean_motion * cos_theta_sin_i)
# plt.plot(t, np.gradient(sin_theta_sin_i))
#
# plt.figure()
# plt.plot(t, -mean_motion * sin_theta_sin_i + np.cos(i) * F_z / a / mean_motion / satellite_mass)
# plt.plot(t, np.gradient(cos_theta_sin_i))

plt.figure()
plt.plot(t, np.gradient(np.cos(true_anomaly) * np.tan(i / 2) - np.cos(true_anomaly + RAAN - RAAN_ref) * np.tan(inc_ref / 2), edge_order=2))
plt.plot(t, F_z / (1 + np.cos(i)) / a / mean_motion / satellite_mass)

plt.show()