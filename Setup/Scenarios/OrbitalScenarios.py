from enum import Enum


class Orbit:
    """
    Very basic class representing an orbit.
    """

    def __init__(self, inclination: float = 0, eccentricity: float = 0, longitude: float = 0,
                 argument_of_periapsis: float = 0):
        """
        Create an orbit using standard orbital elements, except for the radius and anomaly.

        :param inclination: Inclination in deg.
        :param eccentricity: Eccentricity.
        :param longitude: Longitude in deg.
        :param argument_of_periapsis: Argument of periapsis in deg.
        """
        self.inclination = inclination
        self.eccentricity = eccentricity
        self.longitude = longitude
        self.argument_of_periapsis = argument_of_periapsis

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Orbit with {self.inclination=}, {self.eccentricity=}, " \
               f"{self.longitude=} and {self.argument_of_periapsis=}"


class OrbitalScenarios(Enum):
    """
    An Enum for different orbits.
    """
    equatorial_orbit = Orbit()
    tilted_orbit_45deg = Orbit(inclination=45, argument_of_periapsis=10, longitude=20)
    arbitrary_orbit = Orbit(inclination=30, longitude=10)
    eccentric_orbit = Orbit(inclination=45, eccentricity=0.1, argument_of_periapsis=10, longitude=20)


if __name__ == '__main__':
    print(list(OrbitalScenarios))
