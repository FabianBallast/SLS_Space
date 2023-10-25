import numpy as np


def convert_blend2mean(blend_state: np.ndarray):
    """
    Convert a blend state to a mean OE state.

    :param blend_state: Blend state to convert.
    :return: Mean orbital element state.
    """
    r = blend_state[:, 0::6] + 55
    lambdaf = blend_state[:, 1::6]
    ex = blend_state[:, 2::6]
    ey = blend_state[:, 3::6]
    xix = blend_state[:, 4::6]
    xiy = blend_state[:, 5::6]

    e = np.sqrt(ex ** 2 + ey ** 2)
    f = np.zeros_like(e)
    f[e > 0] = np.arctan2(ey[e > 0], ex[e > 0])

    eq_1 = xix / np.tan(np.deg2rad(45) / 2)
    eq_2 = xiy / np.tan(np.deg2rad(45) / 2)
    Omega = np.arccos((eq_1 ** 2 + eq_2 ** 2 - 2) / -2)

    if xiy > 0:
        Omega *= -1

    theta = lambdaf - np.cos(np.deg2rad(45)) * Omega

    eta = np.sqrt(1 - e ** 2)
    kappa = 1 + e * np.cos(f)
    a = r * kappa / eta ** 2
    omega = theta - f

    return np.array([a, e, np.deg2rad(45) * np.ones_like(a), omega, Omega, f])
