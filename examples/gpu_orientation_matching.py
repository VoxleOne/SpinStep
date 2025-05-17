import numpy as np
from spinstep.discrete import DiscreteOrientationSet
from spinstep.utils.quaternion_math import batch_quaternion_angle

N = 100_000
orientations = np.random.randn(N, 4)
orientation_set = DiscreteOrientationSet(orientations, use_cuda=True)

query = np.array([[0, 0, 0, 1]])
xp = orientation_set.xp
angles = batch_quaternion_angle(xp.array(query), orientation_set.orientations, xp)
close_inds = xp.where(angles[0] < 0.1)[0]
print(f"Found {close_inds.size} close orientations (on GPU)")
