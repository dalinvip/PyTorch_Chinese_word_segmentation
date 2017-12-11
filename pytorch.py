# line = "0 simplistic , silly and tedious ."
# label, sep, text = line.partition(' ')
# print(label)
# print(sep)
# print(text)



# from torch.autograd import Variable
# import torch
# a = torch.zeros(18)
# a = Variable(a)
# print(a)
# c = 0
# print(a.numel())
# c += a.numel()
# print(c)
# b = torch.randn(1,2,3,4,5)
# print(b)
# print(b.numel())




# def cal_mean(list):
#     sum = 0.0
#     for i in list:
#         print(i)
#         sum += i
#     avg = (sum / len(list))
#     return avg
#
# list = [10.9, 9.9, 8.8]
# mean = cal_mean(list)
# print(mean)




# import torch
# a = torch.FloatTensor([[1, 2], [3, 4]])
# b = torch.FloatTensor([[1, 2], [3, 4]])
# print(a)
# print(b)
# c = torch.mm(a, b)
# print(c)
# d = a.mm(b)
# print(d)




# from torch.autograd import Function, Variable
# import torch
# class sru_com(Function):
#     def __init__(self,b):
#         print("init")
#
#     def forward(self,c):
#         print("forward")
#         return None
#
# x = sru_com(Variable(torch.randn(3,4)))(Variable(torch.randn(3,3)))


# list = [1, 2, 3, 4, 5, 6]
# print(list[-2::-1])
# print(list[-2::-2])
# print(list[-1::-1])

# import time
# start = time.time()
# for i in range(1000000):
#     print(i * 2)
#     print(i * 3 * 8 / 5 - 4)
# end = time.time()
# print("Times: ", end - start)

import torch
import numpy as np
# a = [2.0, 5, 2.0, 2]
# b = [1.5, 0.1, 1, 1]
# c = np.subtract(a, b)
# print(c)
# print(c[0])

# a = np.random.rand(240000,5)
# print(a)

# a = np.sqrt(0)
# print(a)

# a = torch.rand(3, 4, 5)
# print(a)
# print(a.reverse())
# print(a[0][0])
# list = (a[0][0]).tolist()
# b = list.reverse()
# print(list)
# print(list.reverse())
# b = reversed(list)
# print(b)
# b = torch.inverse(a)
# print(b)



# a = 1 % -1 == 0
# print(a)
#
# b = 4 % -1 == 0
# print(b)


a = [[1, 2, 3], [1, 2, 4]]
print(a)
b = torch.LongTensor(a)
print(b)