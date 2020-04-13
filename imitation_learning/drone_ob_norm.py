"""Observation norm used by behavioral cloning agents and the shared autonomy agent."""
import numpy as np
import gin


@gin.configurable
def drone_bc_mean():
    return np.array([0.0043, -0.0065,  0.0965,  0.7479, -0.0983, -2.4260,
                     -0.0219, -0.0484, 0.3360, -0.0124, -0.0320,  0.0437,
                     0.0000,  0.0000,  0.0570],
                    dtype=np.float32)


@gin.configurable
def drone_bc_std():
    return np.array([0.2092, 0.2249, 0.0903, 4.4870, 4.4999, 4.6344, 0.4893,
                     0.2624, 2.6024, 0.2528, 0.2541, 0.5348, 0.3333, 0.3333,
                     0.0455], dtype=np.float32)
