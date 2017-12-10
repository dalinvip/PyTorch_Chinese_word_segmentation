# coding=utf-8
import torch
from torch.autograd import Variable
import random
from loaddata.common import sep, app, nullkey, paddingkey, unkkey
from loaddata.Instance import instance, Batch_Features
import hyperparams as hy
torch.manual_seed(hy.seed_num)
random.seed(hy.seed_num)


class Iterators():
    # def __init__(self, batch_size=1, data=None, operator=None):
    def __init__(self):
        self.args = None
        self.batch_size = None
        self.data = None
        self.operator = None
        self.operator_static = None
        self.iterator = []
        self.batch = []
        self.features = []
        self.data_iter = []

    def createIterator(self, batch_size=None, data=None, operator=None, operator_static=None, args=None):
        assert isinstance(data, list), "ERROR: data must be in list [train_data,dev_data]"
        assert isinstance(batch_size, list), "ERROR: batch_size must be in list [16,1,1]"
        self.args = args
        self.batch_size = batch_size
        self.data = data
        self.operator = operator
        self.operator_static = operator_static
        for id_data in range(len(data)):
            print("*****************    create {} iterator    **************".format(id_data + 1))
            self.convert_word2id(self.data[id_data], self.operator, self.operator_static)
            self.features = self.create_onedata_Iterator(insts=self.data[id_data], batch_size=self.batch_size[id_data],
                                                         operator=self.operator,
                                                         operator_static=self.operator_static)
            self.data_iter.append(self.features)
            self.features = []
        return self.data_iter[0], self.data_iter[1], self.data_iter[2]
        # return self.data_iter[0]

    def convert_word2id(self, insts, operator, operator_static):
        # print(len(insts))
        # for index_inst, inst in enumerate(insts):
        for inst in insts:
            # copy with the word and pos
            for index in range(inst.words_size):
                word = inst.words[index]
                wordID = operator.word_alphabet.loadWord2idAndId2Word(word)
                # if wordID is None:
                if wordID == -1:
                    wordID = operator.word_UnkkID
                    # wordID = operator
                inst.words_index.append(wordID)

                pos = inst.pos[index]
                posID = operator.pos_alphabet.loadWord2idAndId2Word(pos)
                # if posID is None:
                if posID == -1:
                    posID = operator.pos_UnkID
                inst.pos_index.append(posID)
            # print(inst.words_index)
            # print(inst.pos_index)
            # copy with the char
            for index in range(inst.chars_size):
                char = inst.chars[index]
                charID = operator.char_alphabet.loadWord2idAndId2Word(char)
                static_charID = operator_static.char_alphabet.loadWord2idAndId2Word(char)
                # if charID is None:
                if charID == -1:
                    charID = operator.char_UnkID
                if static_charID == -1:
                    static_charID = operator_static.char_UnkID
                inst.chars_index.append(charID)
                inst.static_chars_index.append(static_charID)
            # print(inst.chars_index)
            # copy with the bichar_left
            for index in range(inst.bichars_size):
                bichar_left = inst.bichars_left[index]
                bichar_left_ID = operator.bichar_alphabet.loadWord2idAndId2Word(bichar_left)
                static_bichar_left_ID = operator_static.bichar_alphabet.loadWord2idAndId2Word(bichar_left)
                # if bichar_left_ID is None:
                if bichar_left_ID == -1:
                    bichar_left_ID = operator.bichar_UnkID
                if static_bichar_left_ID == -1:
                    static_bichar_left_ID = operator_static.bichar_UnkID
                inst.bichars_left_index.append(bichar_left_ID)
                inst.static_bichars_left_index.append(static_bichar_left_ID)
            # print(inst.bichars_left_index)

            # copy with the bichar_right
            for index in range(inst.bichars_size):
                bichar_right = inst.bichars_right[index]
                bichar_right_ID = operator.bichar_alphabet.loadWord2idAndId2Word(bichar_right)
                static_bichar_right_ID = operator_static.bichar_alphabet.loadWord2idAndId2Word(bichar_right)
                # if bichar_right_ID == -1:
                # if bichar_right_ID is None:
                if bichar_right_ID == -1:
                    bichar_right_ID = operator.bichar_UnkID
                if static_bichar_right_ID == -1:
                    static_bichar_right_ID = operator_static.bichar_UnkID
                inst.bichars_right_index.append(bichar_right_ID)
                inst.static_bichars_right_index.append(static_bichar_right_ID)
            # print(inst.bichars_right_index)

            # copy with the gold
            for index in range(inst.gold_size):
                gold = inst.gold[index]
                goldID = operator.label_alphabet.loadWord2idAndId2Word(gold)
                # print("gold ID ", goldID)
                inst.gold_index.append(goldID)

    def create_onedata_Iterator(self, insts, batch_size, operator, operator_static):
        batch = []
        count_inst = 0
        for index, inst in enumerate(insts):
            batch.append(inst)
            count_inst += 1
            # print(batch)
            if len(batch) == batch_size or count_inst == len(insts):
                # print("aaaa", len(batch))
                one_batch = self.create_one_batch(insts=batch, batch_size=batch_size, operator=operator,
                                                  operator_static=operator_static)
                self.features.append(one_batch)
                batch = []
        print("The all data has created iterator.")
        return self.features

    def create_one_batch(self, insts, batch_size, operator, operator_static):
        # print("create one batch......")
        batch_length = len(insts)
        # copy with the max length for padding
        max_word_size = -1
        max_char_size = -1
        max_bichar_size = -1
        max_gold_size = -1
        max_pos_size = -1
        for inst in insts:
            word_size = inst.words_size
            if word_size > max_word_size:
                max_word_size = word_size
            char_size = inst.chars_size
            if char_size > max_char_size:
                max_char_size = char_size
            bichar_size = inst.bichars_size
            if bichar_size > max_bichar_size:
                max_bichar_size = bichar_size
            gold_size = inst.gold_size
            if gold_size > max_gold_size:
                max_gold_size = gold_size

        # create with the Tensor/Variable
        # word features
        batch_word_features = Variable(torch.LongTensor(batch_length, max_word_size))
        batch_pos_features = Variable(torch.LongTensor(batch_length, max_word_size))

        batch_char_features = Variable(torch.LongTensor(batch_length, max_char_size))
        batch_bichar_left_features = Variable(torch.LongTensor(batch_length, max_bichar_size))
        batch_bichar_right_features = Variable(torch.LongTensor(batch_length, max_bichar_size))

        batch_static_char_features = Variable(torch.LongTensor(batch_length, max_char_size))
        batch_static_bichar_left_features = Variable(torch.LongTensor(batch_length, max_bichar_size))
        batch_static_bichar_right_features = Variable(torch.LongTensor(batch_length, max_bichar_size))

        batch_gold_features = Variable(torch.LongTensor(max_gold_size * batch_length))

        # print(batch_gold_features)

        for id_inst in range(batch_length):
            inst = insts[id_inst]
            # print(inst.words_index)
            # copy with the word features
            for id_word_index in range(max_word_size):
                if id_word_index < inst.words_size:
                    batch_word_features.data[id_inst][id_word_index] = inst.words_index[id_word_index]
                else:
                    # print(operator.word_PaddingID)
                    batch_word_features.data[id_inst][id_word_index] = operator.word_PaddingID

            # copy with the pos features
            for id_pos_index in range(max_word_size):
                if id_pos_index < inst.words_size:
                    batch_pos_features.data[id_inst][id_pos_index] = inst.pos_index[id_pos_index]
                else:
                    # print("aaa", operator.pos_PaddingID)
                    batch_pos_features.data[id_inst][id_pos_index] = operator.pos_PaddingID

            # copy with the char features
            for id_char_index in range(max_char_size):
                if id_char_index < inst.chars_size:
                    batch_char_features.data[id_inst][id_char_index] = inst.chars_index[id_char_index]
                    batch_static_char_features.data[id_inst][id_char_index] = inst.static_chars_index[id_char_index]
                else:
                    # print("aaa", operator.char_PaddingID)
                    batch_char_features.data[id_inst][id_char_index] = operator.char_PaddingID
                    batch_static_char_features.data[id_inst][id_char_index] = operator_static.char_PaddingID

            # copy with the bichar_left features
            for id_bichar_left_index in range(max_bichar_size):
                if id_bichar_left_index < inst.bichars_size:
                    batch_bichar_left_features.data[id_inst][id_bichar_left_index] = inst.bichars_left_index[id_bichar_left_index]
                    batch_static_bichar_left_features.data[id_inst][id_bichar_left_index] = int(inst.static_bichars_left_index[id_bichar_left_index])
                else:
                    # print("aaa", operator.bichar_PaddingID)
                    batch_bichar_left_features.data[id_inst][id_bichar_left_index] = operator.bichar_PaddingID
                    batch_static_bichar_left_features.data[id_inst][id_bichar_left_index] = int(operator_static.bichar_PaddingID)

            # copy with the bichar_right features
            for id_bichar_right_index in range(max_bichar_size):
                if id_bichar_right_index < inst.bichars_size:
                    # batch_bichar_right_features.data[id_inst][id_bichar_right_index] = inst.bichars_left_index[id_bichar_right_index]
                    batch_bichar_right_features.data[id_inst][id_bichar_right_index] = inst.bichars_right_index[id_bichar_right_index]
                    batch_static_bichar_right_features.data[id_inst][id_bichar_right_index] = inst.static_bichars_right_index[id_bichar_right_index]
                else:
                    batch_bichar_right_features.data[id_inst][id_bichar_right_index] = operator.bichar_PaddingID
                    batch_static_bichar_right_features.data[id_inst][id_bichar_right_index] = operator_static.bichar_PaddingID

            # copy with the gold features
            for id_gold_index in range(max_gold_size):
                if id_gold_index < inst.gold_size:
                    # print("wwwwww", inst.gold_index[id_gold_index])
                    batch_gold_features.data[id_gold_index + id_inst * max_gold_size] = inst.gold_index[id_gold_index]
                else:
                    batch_gold_features.data[id_gold_index + id_inst * max_gold_size] = 0

        # batch
        features = Batch_Features()
        features.batch_length = batch_length
        features.inst = insts
        features.word_features = batch_word_features
        features.pos_features = batch_pos_features
        features.char_features = batch_char_features
        features.static_char_features = batch_static_char_features
        features.bichar_left_features = batch_bichar_left_features
        features.static_bichar_left_features = batch_static_bichar_left_features
        features.bichar_right_features = batch_bichar_right_features
        features.static_bichar_right_features = batch_static_bichar_right_features
        features.gold_features = batch_gold_features
        if self.args.use_cuda is True:
            features.cuda(features)
        return features

    def padding(self, batch_length, insts, operator=None, batch_features=None, max_size=-1):
        print("padding")
        for id_inst in range(batch_length):
            inst = insts[id_inst]
            for id_word_index in range(max_size):
                if id_word_index < inst.words_size:
                    print("Failed to write a function for all")

        return ""











