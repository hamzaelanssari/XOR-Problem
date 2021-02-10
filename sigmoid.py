import numpy as np

''' Sigmoid Class '''


class sigmoid:
    def function(x):
        return 1 / (1 + np.exp(-x))

    def derivative(x):
        return x * (1 - x)


''' End Sigmoid Class '''
