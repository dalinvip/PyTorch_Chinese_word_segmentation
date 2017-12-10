# coding=utf-8

import os
import re
import torch
import shutil
import random
import unicodedata
from loaddata.Alphabet import Create_Alphabet
from loaddata.Instance import instance
from loaddata.common import sep, app, paddingkey, unkkey, nullkey
# fix the random seed
import hyperparams as hy
torch.manual_seed(hy.seed_num)
random.seed(hy.seed_num)


class load_data():

    def __init__(self, path=[], shuffle=False):
        print("load data for train/dev/test")
        self.debug_index = -1
        self.path = path
        self.date_list = []

    def clean_str(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip()

    def load_data(self, path=None, shuffle=False):
        assert isinstance(path, list), "path must be in list"
        for id_data in range(len(path)):
            print(path[id_data])
            # sort the data with sentence length
            self.sort_data(path=path[id_data])
            # insts = None
            # insts = self.load_one_date(path=path[id_data], shuffle=shuffle)
            insts = self.load_one_date(path="./temp_data.txt", shuffle=False)
            self.date_list.append(insts)
        return self.date_list[0], self.date_list[1], self.date_list[2]

    def sort_data(self, path=None):
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
            lines.sort(key=lambda x: len(x))
            if os.path.exists("./temp_data.txt"):
                os.remove("./temp_data.txt")

        file = open("./temp_data.txt", mode="w", encoding="UTF-8")
        file.writelines(lines)
        # for line in lines:
        #     file.write(line)

    def load_one_date(self, path=None, shuffle=False):
        print("loading {} data......".format(path))
        assert path is not None
        insts = []
        with open(path, encoding="UTF-8") as f:
            lines = f.readlines()
            for index, line in enumerate(lines):
                # copy with "/n"
                line = unicodedata.normalize('NFKC', line.strip())
                # init instance
                inst = instance()
                line = line.split(" ")
                count = 0
                for word_pos in line:
                    # segment the word and pos in line
                    word, _, label = word_pos.partition("_")
                    word_length = len(word)
                    inst.words.append(word)
                    inst.gold_seg.append("[" + str(count) + "," + str(count + word_length) + "]")
                    inst.gold_pos.append("[" + str(count) + "," + str(count + word_length) + "]" + label)
                    count += word_length
                    for i in range(word_length):
                        char = word[i]
                        # print(char)
                        inst.chars.append(char)
                        if i == 0:
                            inst.gold.append(sep + "#" + label)
                            inst.pos.append(label)
                        else:
                            inst.gold.append(app)
                char_number = len(inst.chars)
                for i in range(char_number):
                    # copy with the left bichars
                    if i is 0:
                        inst.bichars_left.append(nullkey + inst.chars[i])
                    else:
                        inst.bichars_left.append(inst.chars[i - 1] + inst.chars[i])
                    # copy with the right bichars
                    if i == char_number - 1:
                        inst.bichars_right.append(inst.chars[i] + nullkey)
                    else:
                        inst.bichars_right.append(inst.chars[i] + inst.chars[i + 1])
                # char/word size
                inst.chars_size = len(inst.chars)
                inst.words_size = len(inst.words)
                inst.bichars_size = len(inst.bichars_left)
                inst.gold_size = len(inst.gold)
                # add one inst that represent one sentence into the list
                insts.append(inst)
                if index == self.debug_index:
                    break
        if shuffle is True:
            print("shuffle tha data......")
            random.shuffle(insts)
        # return all sentence in data
        # print(insts[-1].words)
        return insts


if __name__ == "__main__":
    print("Test dataloader........")
    loaddata = load_data()
    # train_data, dev_data, test_data = loaddata.load_data(path=("../pos_test_data/train.ctb60.pos-1.hwc", "../pos_test_data/dev.ctb60.pos.hwc",
    #                             "../pos_test_data/test.ctb60.pos.hwc"), shuffle=True)
    train_data, dev_data, test_data = loaddata.load_data(path=["../pos_test_data/train.ctb60.pos-1.hwc", "../pos_test_data/train.ctb60.pos-1.hwc"
        , "../pos_test_data/train.ctb60.pos-1.hwc"], shuffle=True)
    # print("\n\n\n\n")
    # print(train_data)
    # print("\n\n\n\n")
    # print(dev_data)
    # print("\n\n\n\n")
    # print(test_data)
    # create_alphabet = Create_Alphabet(min_freq=1)
    # create_alphabet.createAlphabet(train_data=train_data, dev_data=dev_data, test_data=test_data, debug_index=-1)
    # load_data.loaddate("../pos_test_data/test.ctb60.pos.hwc")
