"""
SVPG test procedure, use normal distribution as target distribution.
We want svpg generate params as close as target distribution sampling results.
All params range from -1 to 1.
We define Loss func as 'MLE' between svpg output and target output.
"""

import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal

dimension = 5
mean_vector = [np.float64(i) for i in range(dimension)]

# Get target distribution
target_distribution = MultivariateNormal(torch.tensor(mean_vector), torch.eye(dimension))
y = target_distribution.sample()
print(y)

# Get random input
x = np.random.uniform(-1, 1, dimension)
print(x, x.dtype)
