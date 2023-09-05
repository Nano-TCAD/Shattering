import numpy as np


class Grid(object):

    def __init__(self, n: int, gamma: float):
        self.gamma = gamma
        self.omega = (gamma ** 4) / (4 *  n ** 5)   
        self.size = int(np.ceil(8 / self.omega))

        # Select from a uniform distribution a point in [-4, -4 + w) x [-4, -4 + w)
        rng = np.random.default_rng()
        x = rng.uniform(-4, -4 + self.omega)
        y = rng.uniform(-4, -4 + self.omega)
        self.z = x + 1j * y

    def __repr__(self):
        return f"Grid(z={self.z}, omega={self.omega}, size={self.size})"
