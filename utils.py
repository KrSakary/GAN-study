import numpy as np

import hparam

# Generate noise used for generator
def z_noise():
    z = np.random.uniform(low=0, high=1, size=[hparam.batch_size, hparam.noise_dim])

    return z