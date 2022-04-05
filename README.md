# SVPG-Pytorch
Stein Variance Policy Gradients Algorithm Pytorch implementation

2022-4-5

- inplace operation 错误，修改torch版本为1.0.0后无报错；并且修改两次backward均为retain_graph=True。