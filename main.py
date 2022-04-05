"""
SVPG test procedure, use normal distribution as target distribution.
We want svpg generate params as close as target distribution sampling results.
All params range from -1 to 1.
We define Loss func as 'MLE' between svpg output and target output.
"""

import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.uniform import Uniform
from torch.distributions.kl import kl_divergence
from svpg.svpg import SVPG

dimension = 1
n = 100
mean_vector = [np.float32(i * 0.2) for i in range(dimension)]

target_distribution = MultivariateNormal(torch.tensor(mean_vector), torch.eye(dimension) * 0.1)
uniform = Uniform(2, 3)

# Get target distribution
y = target_distribution.rsample((n,))
y2 = uniform.rsample((n,))
# print(y, y.shape, y.dtype)


# Get random input
x = uniform.sample((n, dimension))
# print(x, x.shape, x.dtype)


# Calculate difference of my dist and target dist samples
def distances_between_two_samples(x, y):
    diff = 0.
    assert len(x) == len(y)
    for _x, _y in zip(x, y):
        diff += (_x - _y).pow(2).sum()
    diff /= len(x)
    return diff


diff_x_y = distances_between_two_samples(x, y)
diff_x_y2 = distances_between_two_samples(x, y2)
print("Difference between x and y is: {0}".format(diff_x_y))
print("Difference between x and y is: {0}".format(diff_x_y2))


# TODO: 目标是调用SVPG模块，生成样本与目标分布生成的样本越接近越好

episode = 200
nagents = 10
nparams = dimension
max_step_length = 0.05
svpg_rollout_length = 5
svpg_horizon = 25
temperature = 10
discrete = False
kld_coefficient = 0

svpg_trainer = SVPG(
    nagents=nagents,
    nparams=nparams,
    max_step_length=max_step_length,
    svpg_rollout_length=svpg_rollout_length,
    svpg_horizon=svpg_horizon,
    temperature=temperature,
    discrete=discrete,
    kld_coefficient=kld_coefficient
)

for i in range(episode):
    # y = target_distribution.rsample((nagents, svpg_rollout_length,))
    y = uniform.rsample((nagents, svpg_rollout_length, dimension))
    y = np.array(y)
    x = svpg_trainer.step()
    loss = []
    # print(y.shape, x.shape)
    for x1, y1 in zip(x, y):
        tmp = []
        for x2, y2 in zip(x1, y1):
            diff = np.sum(np.square(x2 - y2))
            tmp.append(diff)
        loss.append(tmp)
    loss = np.array(loss)
    print(loss.mean())
    svpg_trainer.train1(loss)

