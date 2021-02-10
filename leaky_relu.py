import numpy as np

''' Leaky Relu Class '''


class leaky_relu:
    def function(x):
        return np.where(x > 0, x, 0.01 * x)

    def derivative(x):
        return np.where(x > 0, 1, 0.01)


''' End Leaky Relu Class '''
