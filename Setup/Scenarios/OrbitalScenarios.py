equatorial_orbit = {'orbital_height': 750e3,
                    'inclination': 0,
                    'eccentricity': 0,
                    'longitude': 0,
                    'argument_of_periapsis': 0}

tilted_orbit_45deg = {'orbital_height': 750e3,
                      'inclination': 45,
                      'eccentricity': 0,
                      'longitude': 0,
                      'argument_of_periapsis': 0}

equatorial_orbit_scaled = {'orbital_height': 55,
                           'inclination': 0,
                           'eccentricity': 0,
                           'longitude': 0,
                           'argument_of_periapsis': 0}

tilted_orbit_45deg_scaled = {'orbital_height': 55,
                             'inclination': 45,
                             'eccentricity': 0,
                             'longitude': 0,
                             'argument_of_periapsis': 0}

orbital_scenarios = {'equatorial_orbit': equatorial_orbit,
                     'tilted_orbit_45deg': tilted_orbit_45deg,
                     'equatorial_orbit_scaled': equatorial_orbit_scaled,
                     'tilted_orbit_45deg_scaled': tilted_orbit_45deg_scaled}


def print_orbital_scenarios() -> None:
    """
    Print the available scenarios regarding orbital parameters.
    """
    print(f"The available orbital scenarios are: {list(orbital_scenarios.keys())}")


if __name__ == '__main__':
    print_orbital_scenarios()
