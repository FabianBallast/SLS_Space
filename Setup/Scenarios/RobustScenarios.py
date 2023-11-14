from enum import Enum


class RobustnessType(Enum):
    """
    Class to represent type of robustness approach.
    """
    NO_ROBUSTNESS = 1
    SIMPLE_ROBUSTNESS = 2
    ADVANCED_ROBUSTNESS = 3


class Robustness:
    """
    Class to represent different robustness approaches with different values.
    """
    def __init__(self, e_A: float = None, e_B: float = None, e_A0: float = None, e_A1: float = None, e_A2: float = None,
                 e_A3: float = None, e_A4: float = None, e_A5: float = None, e_B0: float = None, e_B1: float = None,
                 e_B2: float = None, e_B3: float = None, e_B4: float = None, e_B5: float = None, sigma_w: float = None,
                 sigma_w0: float = None, sigma_w1: float = None, sigma_w2: float = None, sigma_w3: float = None,
                 sigma_w4: float = None, sigma_w5: float = None):

        self.e_A = e_A
        self.e_B = e_B
        self.sigma_w = sigma_w

        self.e_A0 = e_A0
        self.e_A1 = e_A1
        self.e_A2 = e_A2
        self.e_A3 = e_A3
        self.e_A4 = e_A4
        self.e_A5 = e_A5

        self.e_B0 = e_B0
        self.e_B1 = e_B1
        self.e_B2 = e_B2
        self.e_B3 = e_B3
        self.e_B4 = e_B4
        self.e_B5 = e_B5

        self.sigma_w0 = sigma_w0
        self.sigma_w1 = sigma_w1
        self.sigma_w2 = sigma_w2
        self.sigma_w3 = sigma_w3
        self.sigma_w4 = sigma_w4
        self.sigma_w5 = sigma_w5

        if self.e_A is None and self.e_B is None and self.e_A0 is None and self.e_B0 is None:
            self.robustness_type = RobustnessType.NO_ROBUSTNESS
        elif self.e_A0 is None and self.e_B0 is None:
            self.robustness_type = RobustnessType.SIMPLE_ROBUSTNESS
        else:
            self.robustness_type = RobustnessType.ADVANCED_ROBUSTNESS


class RobustnessScenarios(Enum):
    """
    Simple Enum for different robustness scenarios.
    """
    no_robustness = Robustness()
    # no_robustness = Robustness(e_A=0, e_B=0, sigma_w=0)
    # no_robustness = Robustness(e_A0=0, e_A1=0, e_A2=0, e_A3=0, e_A4=0,
    #                            e_A5=0,  e_B0=0, e_B1=0, e_B2=0,
    #                            e_B3=0, e_B4=0, e_B5=0, sigma_w=0)

    # simple_robustness_no_noise = Robustness(e_A=0.0020, e_B=0.0022, sigma_w=0)
    # advanced_robustness_no_noise = Robustness(e_A0=0.00059, e_A1=0.0020, e_A2=0.000027, e_A3=0.000048, e_A4=0.0013,
    #                                       e_A5=0.00087,  e_B0=0.00030, e_B1=0.0022, e_B2=0.000081,
    #                                       e_B3=0.00013, e_B4=0.00059, e_B5=0.00099, sigma_w=0)
    #
    # simple_robustness_noise = Robustness(e_A=0.0020, e_B=0.0022, sigma_w=0.001)
    # advanced_robustness_noise = Robustness(e_A0=0.00059, e_A1=0.0020, e_A2=0.000027, e_A3=0.000048, e_A4=0.0013,
    #                                           e_A5=0.00087, e_B0=0.00030, e_B1=0.0022, e_B2=0.000081,
    #                                           e_B3=0.00013, e_B4=0.00059, e_B5=0.00099, sigma_w=0.001)

    # simple_robustness_no_noise = Robustness(e_A=0.012, e_B=0.033, sigma_w=0)
    # advanced_robustness_no_noise = Robustness(e_A0=0.011, e_A1=0.013, e_A2=0.00024, e_A3=0.00018, e_A4=0.0046,
    #                                           e_A5=0.0069, e_B0=0.033, e_B1=0.020, e_B2=0.0018,
    #                                           e_B3=0.0021, e_B4=0.0085, e_B5=0.0028, sigma_w=0)

    # 10 s
    simple_robustness_no_noise = Robustness(e_A=0.0021, e_B=0.0023, sigma_w=0)
    advanced_robustness_no_noise = Robustness(e_A0=0.0020, e_A1=0.0021, e_A2=0.0000286, e_A3=0.000496, e_A4=0.0015,
                                              e_A5=0.00086, e_B0=0.00031, e_B1=0.0023, e_B2=0.000081,
                                              e_B3=0.00013, e_B4=0.00060, e_B5=0.00091, sigma_w0=0, sigma_w1=0,
                                              sigma_w2=0, sigma_w3=0, sigma_w4=0, sigma_w5=0)
    # advanced_robustness_no_noise = Robustness(e_A0=0.0021, e_A1=0.0021, e_A2=0.0021, e_A3=0.0021, e_A4=0.0021,
    #                                           e_A5=0.0021, e_B0=0.0021, e_B1=0.0023, e_B2=0.0023,
    #                                           e_B3=0.0023, e_B4=0.0023, e_B5=0.0023, sigma_w0=0.001, sigma_w1=0.001,
    #                                           sigma_w2=0.001, sigma_w3=0.001, sigma_w4=0.001, sigma_w5=0.001)



    # 20 s
    # simple_robustness_no_noise = Robustness(e_A=0.0041, e_B=0.0049, sigma_w=0)
    # advanced_robustness_no_noise = Robustness(e_A0=0.0017, e_A1=0.0041, e_A2=0.000057, e_A3=0.00011, e_A4=0.0030,
    #                                           e_A5=0.0021, e_B0=0.0017, e_B1=0.0049, e_B2=0.00026,
    #                                           e_B3=0.00040, e_B4=0.0016, e_B5=0.0021, sigma_w=0)

    simple_robustness_noise = Robustness(e_A=0.0021, e_B=0.0023, sigma_w=0.0017)
    advanced_robustness_noise = Robustness(e_A0=0.0020, e_A1=0.0021, e_A2=0.0000286, e_A3=0.0000496, e_A4=0.0015,
                                           e_A5=0.00086, e_B0=0.00031, e_B1=0.0023, e_B2=0.000081,
                                           e_B3=0.00013, e_B4=0.00060, e_B5=0.00091, sigma_w0=0.0016, sigma_w1=0.0017,
                                           sigma_w2=0.00011, sigma_w3=0.00011, sigma_w4=0.00093, sigma_w5=0.00088)


