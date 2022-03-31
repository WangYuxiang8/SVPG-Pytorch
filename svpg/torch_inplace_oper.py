# Learning scripts
# Pytorch inplace operation
# See details in: https://zhuanlan.zhihu.com/p/38475183

import torch


def error_1():
    x = torch.FloatTensor([[1., 2.]])
    w1 = torch.FloatTensor([[2.], [1.]])
    w2 = torch.FloatTensor([3.])
    w1.requires_grad = True
    w2.requires_grad = True

    d = torch.matmul(x, w1)
    f = torch.matmul(d, w2)
    # 因为这句, 代码报错了 RuntimeError: one of the variables needed for gradient computation has been modified by an
    # inplace operation
    d[:] = 1

    f.backward()


def error_2():
    import torch
    x = torch.FloatTensor([[1., 2.]])
    w1 = torch.FloatTensor([[2.], [1.]])
    w2 = torch.FloatTensor([3.])
    w1.requires_grad = True
    w2.requires_grad = True

    d = torch.matmul(x, w1)

    d_ = d.data

    f = torch.matmul(d, w2)
    d_[:] = 1

    f.backward()

    print(w1.grad, w2.grad, x.grad)


def correct_1():
    x = torch.FloatTensor([[1., 2.]])
    w1 = torch.FloatTensor([[2.], [1.]])
    w2 = torch.FloatTensor([3.])
    w1.requires_grad = True
    w2.requires_grad = True

    d = torch.matmul(x, w1)
    d[:] = 1  # 稍微调换一下位置, 就没有问题了
    f = torch.matmul(d, w2)
    f.backward()

    print(w1.grad, w2.grad, x.grad)


def correct_2():
    x = torch.FloatTensor([[1., 2.]])
    w1 = torch.FloatTensor([[2.], [1.]])
    w2 = torch.FloatTensor([3.])
    w1.requires_grad = True
    w2.requires_grad = True

    d = torch.matmul(x, w1)
    f = torch.matmul(d, w2)
    f.backward()

    print(w1.grad, w2.grad, x.grad)


def correct_3():
    import torch
    x = torch.FloatTensor([[1., 2.]])
    w1 = torch.FloatTensor([[2.], [1.]])
    w2 = torch.FloatTensor([3.])
    w1.requires_grad = True
    w2.requires_grad = True

    d = torch.matmul(x, w1)

    d_ = d.detach()  # 换成 .detach(), 就可以看到 程序报错了...

    f = torch.matmul(d_, w2)
    # d_[:] = 1
    f.backward()

try:
    error_1()
except:
    print("error.")
error_2()
correct_1()
correct_2()
correct_3()
