import numpy as np
import control as ct


class KalmanFilter:

    def __init__(self, sampling_time: int | float, earth_radius: int | float, mu: int | float, j2_constant: int | float,
                 controller_sampling_time: int | float, satellite_mass: int | float):
        """
        Initialise the Kalman filter.

        :param sampling_time: Sampling time of the simulation.
        :param earth_radius: Radius of the Earth in m.
        :param mu: Gravitational parameter of the Earth.
        :param j2_constant: Value of the J2 constant.
        :param controller_sampling_time: Sampling time of the controller.
        :param satellite_mass: Mass of the satellite in kg.
        """
        self.state = None
        self.cov_mat = None
        self.sampling_time = sampling_time
        self.earth_radius = earth_radius
        self.j2_constant = j2_constant
        self.mu = mu
        self.controller_sampling_time = controller_sampling_time
        self.satellite_mass = satellite_mass

        if self.j2_constant > 0.000001:
            self.Q = [1e4, 1e-0, 0.3e-6, 1e-7]   #  [0.5e-5, 1e-6, 0.5e-7, 1e-8]
        else:
            self.Q = [10000, 10000, 10000, 10000]
        self.R = [2.76e2, 7.74e-2, 2.35e-2, 14.87e-2]

    # def set_measurement_covariance(self, covariance_matrix: np.ndarray) -> None:
    #     """
    #     Set the matrix that represents measurement noise.
    #
    #     :param covariance_matrix: The matrix with the covariance variables.
    #     """
    #     self.R = covariance_matrix]

    def initialise_filter(self, state: np.ndarray) -> None:
        """
        Initialise the filter by setting the initial state variables.

        :param state: Initial state in the shape (4 * number_of_satellites, 1).
        """
        self.state = state
        number_of_satellites = state.shape[0] // 4
        diag_elem = [self.Q[0] / self.R[0], self.Q[1] / self.R[1], self.Q[2] / self.R[2], self.Q[3] / self.R[3]]

        self.cov_mat = np.diag(diag_elem * number_of_satellites)

    def filter_data(self, measurements: np.ndarray, inputs: np.ndarray, update_state: bool = True,
                    inputs_with_same_sampling_time: bool = False) -> np.ndarray:
        """
        Filter the measurements to ignore oscillations.

        :param measurements: The measurements that need to be filtered in the shape (t, 6 * number_of_satellites).
        :param inputs: The inputs that are applied in the shape (t, 3 * number_of_satellites).
        :param update_state: Update the state and covariance matrix variables for the next call.
        :param inputs_with_same_sampling_time: Whether the inputs have the same sampling time.
        :return: The filtered measurements in the shape (t, 6 * number_of_satellites).
        """
        state_copy = np.copy(self.state)
        cov_copy = np.copy(self.cov_mat)

        filtered_data = np.zeros_like(measurements)

        # Find Mean anomaly
        semi_major_axis = measurements[:, 0::6]
        eccentricity = measurements[:, 1::6]
        inclination = measurements[:, 2::6]
        periapsis = measurements[:, 3::6]
        RAAN = measurements[:, 4::6]
        true_anomaly = measurements[:, 5::6]
        mean_anomaly = true_anomaly - 2 * eccentricity * np.sin(true_anomaly)

        # Setup
        number_of_satellites = measurements.shape[1] // 6
        Q = np.diag(number_of_satellites * self.Q)
        R = np.diag(number_of_satellites * self.R)

        mean_motion = np.sqrt(self.mu / semi_major_axis**3)
        j2_scaling_factor = 0.75 * self.j2_constant * (self.earth_radius / (semi_major_axis * (1 - eccentricity ** 2))) ** 2 * mean_motion
        M_dot_j2 = j2_scaling_factor * np.sqrt(1 - eccentricity ** 2) * (3 * np.cos(inclination) ** 2 - 1)
        omega_dot_j2 = j2_scaling_factor * (5 * np.cos(inclination) ** 2 - 1)
        Omega_dot_j2 = j2_scaling_factor * -2 * np.cos(inclination)

        # t_arr = np.arange(0, semi_major_axis.shape[0]).reshape((-1, 1)) * self.sampling_time
        # meas = np.concatenate((semi_major_axis.reshape((-1, 1)), eccentricity.reshape((-1, 1)),
        #                        inclination.reshape((-1, 1)), (RAAN - Omega_dot_j2 * t_arr).reshape((-1, 1))), axis=1)
        # print(np.var(meas, axis=0))


        mean_anomaly_halfway = mean_anomaly + (mean_motion + M_dot_j2) * self.sampling_time / 2
        true_anomaly_halfway = mean_anomaly_halfway + 2 * eccentricity * np.sin(mean_anomaly_halfway)
        periapsis_halfway = periapsis + omega_dot_j2 * self.sampling_time / 2
        eta = np.sqrt(1-eccentricity**2)
        kappa = 1 + eccentricity * np.cos(true_anomaly_halfway)

        K = (self.cov_mat + Q) @ np.linalg.inv(self.cov_mat + Q + R)
        measurement = np.array([semi_major_axis[0], eccentricity[0], inclination[0], RAAN[0]]).T.reshape((-1,))
        state_0 = (np.eye(4 * number_of_satellites) - K) @ self.state + K @ measurement
        filtered_data[0] = np.concatenate((state_0[0::4], state_0[1::4], state_0[2::4], periapsis[0], state_0[3::4],
                                           true_anomaly[0])).reshape((6, -1)).T.flatten()

        for t in range(0, measurements.shape[0] - 1):

            A = np.zeros((4 * number_of_satellites, 4 * number_of_satellites))
            B = np.zeros((4 * number_of_satellites, 3 * number_of_satellites))

            for sat in range(number_of_satellites):
                B[sat * 4: (sat+1) * 4, sat * 3: (sat + 1) * 3] = \
                    np.array([[2 * eccentricity[t, sat] * np.sin(true_anomaly_halfway[t, sat]) * semi_major_axis[t, sat] / eta[t, sat], 2 * kappa[t, sat] * semi_major_axis[t, sat] / eta[t, sat], 0],
                              [eta[t, sat] * np.sin(true_anomaly_halfway[t, sat]), (eta[t, sat] * (1 + kappa[t, sat]) * np.cos(true_anomaly_halfway[t, sat]) + eccentricity[t, sat]) / kappa[t, sat], 0],
                              [0, 0, eta[t, sat] * np.cos(true_anomaly_halfway[t, sat] + periapsis_halfway[t, sat]) / kappa[t, sat]],
                              [0, 0, eta[t, sat] * np.sin(true_anomaly_halfway[t, sat] + periapsis_halfway[t, sat]) / kappa[t, sat] / np.sin(inclination[t, sat])]]) / mean_motion[t, sat] / semi_major_axis[t, sat] / self.satellite_mass

            system_continuous = ct.ss(A, B, np.eye(4 * number_of_satellites), 0)

            # Find discrete system
            system_discrete = ct.sample_system(system_continuous, self.sampling_time)
            A = system_discrete.A
            B = system_discrete.B

            # Filter
            state_change = (np.array([0, 0, 0, 1]) * Omega_dot_j2[t:t+1].T).reshape((-1, )) * self.sampling_time

            if inputs_with_same_sampling_time:
                input_index = t + 1
            else:
                input_index = int((t * self.sampling_time + .0001) / self.controller_sampling_time)

            # print(input_index)
            measurement = np.array([semi_major_axis[t+1], eccentricity[t+1], inclination[t+1], RAAN[t+1]]).T.reshape((-1,))

            # Time update
            state = A @ (self.state + state_change) + B @ inputs[input_index]
            cov_mat = A @ self.cov_mat @ A.T + Q
            state[1::4] = np.maximum(state[1::4], 0)

            # Measurement update
            K = cov_mat @ np.linalg.inv(cov_mat + R)
            self.state = (np.eye(A.shape[0]) - K) @ state + K @ measurement
            self.cov_mat = (np.eye(A.shape[0]) - K) @ cov_mat

            filtered_data[t+1] = np.concatenate((self.state[0::4], self.state[1::4], self.state[2::4],
                                               periapsis[t+1], self.state[3::4],
                                               true_anomaly[t+1])).reshape((6, -1)).T.flatten()

        if not update_state:
            self.state = state_copy
            self.cov_mat = cov_copy

        return filtered_data
