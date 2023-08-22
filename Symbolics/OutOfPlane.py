from sympy import symbols, simplify, diff, tan, cos, sin, sqrt

a_d, r_d, Omega_d, theta_d, i_d, e_d, f_d = symbols("r_d r_d Omega_d theta_d i_d e_d f_d")
a_c, r_c, Omega_c, theta_c, i_c, e_c = symbols("r_c r_c Omega_c theta_c i_c e_c")
u_r, u_t, u_n = symbols("u_r u_t u_n")
mu, theta_1_dot = symbols("mu theta_1_dot")
x1_d = Omega_d + theta_d

x5_d = tan(i_d/2) * cos(Omega_d)
x5_c = tan(i_c/2) * cos(Omega_c)

x6_d = tan(i_d/2) * sin(Omega_d)
x6_c = tan(i_c/2) * sin(Omega_c)

xi5 = simplify(cos(x1_d) * (x5_d - x5_c) + sin(x1_d) * (x6_d-x6_c))

n_d = sqrt(mu / a_d**3)
n_c = sqrt(mu / a_c**3)

i_d_dot = sqrt(1-e_d**2) * cos(theta_d) * u_n / a_d / n_d / (1 + e_d * cos(f_d))
Omega_d_dot = sqrt(1-e_d**2) * sin(theta_d) * u_n / a_d / n_d / (1 + e_d * cos(f_d)) / sin(i_d)
theta_d_dot = sqrt(mu * (1 + e_d * cos(f_d)) / r_d**3) - sqrt(1-e_d**2) * sin(theta_d) * u_n / a_d / n_d / (1 + e_d * cos(f_d)) / tan(i_d)

i_d_dot_simple = cos(theta_d) * u_n
Omega_d_dot_simple = sin(theta_d) * u_n / sin(i_d)
theta_d_dot_simple = theta_1_dot - sin(theta_d) / tan(i_d)

xi5_dot = diff(xi5, i_d) * i_d_dot + diff(xi5, Omega_d) * Omega_d_dot + diff(xi5, theta_d) * theta_d_dot

print(xi5)
print(simplify(diff(xi5, i_d) * i_d_dot_simple))
print(simplify(diff(xi5, Omega_d) * Omega_d_dot_simple))
print(simplify(diff(xi5, theta_d) * theta_d_dot_simple))

print(simplify(diff(xi5, i_d) * i_d_dot_simple + diff(xi5, Omega_d) * Omega_d_dot_simple + diff(xi5, theta_d) * theta_d_dot_simple))
