import numpy as np

''' Softmax Class '''


class softmax:
    def function(x):
        exps = np.exp(x - np.max(x))
        return exps / exps.sum(axis=0, keepdims=True)

    def derivative(x):
        # s = x.reshape((-1, 1))
        # jacobian = np.diagflat(x) - np.dot(s, s.T)
        # d_softmax = jacobian.sum(axis=1)
        return x * (1 - x)


''' End Softmax Class '''
