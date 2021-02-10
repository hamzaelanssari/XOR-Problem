import numpy as np

''' Step Class '''


class step:
    def function(x):
        return np.where(x < 0, 0, 1)

    def derivative(x):
        return 0


''' End Step Class '''
