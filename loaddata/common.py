# coding=utf-8
import torch
import random
import hyperparams as hy
torch.manual_seed(hy.seed_num)
random.seed(hy.seed_num)

unkkey = '-unk-'
nullkey = '-NULL-'
paddingkey = '-padding-'
sep = 'SEP'
app = 'APP'
label = "NN"