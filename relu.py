import numpy as np


''' Relu Class '''


class relu:
    def function(x):
        return np.where(x > 0, x, 0)

    def derivative(x):
        return np.where(x > 0, 1, 0)


''' End Relu Class '''

