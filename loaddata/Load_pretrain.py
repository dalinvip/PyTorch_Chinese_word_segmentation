# coding=utf-8
"""
load pretrain
"""

import torch
import torch.nn.init as init
import numpy as np
import random
import torch.nn as nn
import hyperparams as hy
torch.manual_seed(hy.seed_num)
random.seed(hy.seed_num)


class Load_Pretrain:

    def __init__(self):
        print("loading pretrain")

    def load_pretrain(self, file, alpha, unk, padding):
        f = open(file, encoding='utf-8')
        allLines = f.readlines()
        # allLines = f.readlines()[1:]
        indexs = set()
        info = allLines[0].strip().split(' ')
        embDim = len(info) - 1
        emb = nn.Embedding(alpha.m_size, embDim)

        init.uniform(emb.weight, a=-np.sqrt(3 / embDim), b=np.sqrt(3 / embDim))
        oov_emb = torch.zeros(1, embDim).type(torch.FloatTensor)

        for line in allLines:
            info = line.split(' ')
            wordID = alpha.loadWord2idAndId2Word(info[0])
            if wordID >= 0:
                indexs.add(wordID)
                for idx in range(embDim):
                    val = float(info[idx + 1])
                    emb.weight.data[wordID][idx] = val
                    oov_emb[0][idx] += val
        f.close()
        count = len(indexs) + 1
        for idx in range(embDim):
            oov_emb[0][idx] /= count
        unkID = alpha.loadWord2idAndId2Word(unk)
        paddingID = alpha.loadWord2idAndId2Word(padding)
        for idx in range(embDim):
            emb.weight.data[paddingID][idx] = 0
        # print('UNK ID: ', unkID)
        # print('Padding ID: ', paddingID)
        if unkID != -1:
            for idx in range(embDim):
                emb.weight.data[unkID][idx] = oov_emb[0][idx]
        print("Load Embedding file: ", file, ", size: ", embDim)
        oov = 0
        for idx in range(alpha.m_size):
            if idx not in indexs:
                oov += 1
        print("oov: ", oov, " total: ", alpha.m_size, "oov ratio: ", oov / alpha.m_size)
        print("oov ", unk, "use avg value initialize")
        return emb, embDim


