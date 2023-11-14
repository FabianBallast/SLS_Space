import numpy as np
import matplotlib.pyplot as plt


def predict_Omega(known_values: np.ndarray, direction: np.ndarray=None, time_steps: int=1, deg=3) -> np.ndarray:
    """
    Given a series of Omegas, predict the next value.

    :param known_values: Known values of the series in shape (t, number_of_predictions).
    :param direction: If given, use this direction for the prediction in shape (number_of_predictions).
    :param time_steps: How many time steps to predict.
    :return: Full array with new prediction.
    """
    if direction is None:
        poly_coef = np.polyfit(np.arange(known_values.shape[0]), known_values, deg=deg)
        new_values = np.polyval(poly_coef, known_values.shape[0]).reshape((1, -1))

        return np.concatenate((known_values, new_values))
        # plt.figure()
        # plt.plot(known_values, '-o')
        # plt.plot([known_values.shape[0]] * known_values.shape[1], new_values, 'o')
        # plt.show()
    else:
        time = np.arange(1, time_steps).reshape((-1, 1))
        new_values = np.ones((len(time), known_values.shape[1])) * np.deg2rad(0.1) * time * direction + known_values
        return np.concatenate((known_values, new_values))

if __name__ == '__main__':
    # known_values = np.array([[-7.00067691, -6.9021436, -6.7803054, -6.6414130, -6.4929876, -6.34358035], #-6.20233075
    #                         [-4.00114056, -3.85218747, -3.71238236, -3.59163236, -3.49947548, -3.44390215]]).T  #-3.42883926
    # print(predict_Omega(known_values))

    print(predict_Omega(np.array([[5, -4]]), direction=np.array([-1, 1]), time_steps=4))