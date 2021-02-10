import numpy as np

''' Tanh Class '''
class tanh:
    def function(x):
        t = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return t

    def derivative(x):
        return 1 - x ** 2
''' End Tanh Class '''
