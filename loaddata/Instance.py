# coding=utf-8
import torch
import random
import hyperparams as hy
torch.manual_seed(hy.seed_num)
random.seed(hy.seed_num)

"""
   init instance
"""


class instance():

    def __init__(self):
        # print("init instance......")

        self.words = []
        self.words_size = 0
        self.chars = []
        self.chars_size = 0
        self.bichars_left = []
        self.bichars_right = []
        self.bichars_size = 0

        self.gold = []
        self.pos = []
        self.gold_pos = []
        self.gold_seg = []
        self.gold_size = 0

        self.words_index = []
        self.chars_index = []
        self.bichars_left_index = []
        self.bichars_right_index = []
        self.static_chars_index = []
        self.static_bichars_left_index = []
        self.static_bichars_right_index = []
        self.pos_index = []
        self.gold_index = []


class Batch_Features:
    def __init__(self):

        self.batch_length = 0
        self.inst = None
        self.word_features = 0
        self.pos_features = 0
        self.char_features = 0
        self.bichar_left_features = 0
        self.bichar_right_features = 0
        self.static_char_features = 0
        self.static_bichar_left_features = 0
        self.static_bichar_right_features = 0
        self.gold_features = 0

    def cuda(self, features):
        features.word_features = features.word_features.cuda()
        features.pos_features = features.pos_features.cuda()
        features.char_features = features.char_features.cuda()
        features.static_char_features = features.static_char_features.cuda()
        features.bichar_left_features = features.bichar_left_features.cuda()
        features.static_bichar_left_features = features.static_bichar_left_features.cuda()
        features.bichar_right_features = features.bichar_right_features.cuda()
        features.static_bichar_right_features = features.static_bichar_right_features.cuda()
        features.gold_features = features.gold_features.cuda()

