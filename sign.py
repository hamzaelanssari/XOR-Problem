import numpy as np

''' Sign Class '''


class sign:
    def function(x):
        return np.where(x < 0, -1, 1)

    def derivative(x):
        return 0


''' End Sign Class '''
