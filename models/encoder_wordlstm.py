# coding=utf-8
from loaddata.common import paddingkey
import torch.nn
import torch.nn as nn
import  torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
import numpy as np
import random
import hyperparams as hy
torch.manual_seed(hy.seed_num)
random.seed(hy.seed_num)

"""
    sequence to sequence Encode model
"""


class Encoder_WordLstm(nn.Module):

    def __init__(self, args):
        print("Encoder model --- LSTM")
        super(Encoder_WordLstm, self).__init__()
        self.args = args

        # random
        self.char_embed = nn.Embedding(self.args.embed_char_num, self.args.embed_char_dim)
        for index in range(self.args.embed_char_dim):
            self.char_embed.weight.data[self.args.create_alphabet.char_PaddingID][index] = 0
        self.char_embed.weight.requires_grad = True

        self.bichar_embed = nn.Embedding(self.args.embed_bichar_num, self.args.embed_bichar_dim)
        for index in range(self.args.embed_bichar_dim):
            self.bichar_embed.weight.data[self.args.create_alphabet.bichar_PaddingID][index] = 0
        self.bichar_embed.weight.requires_grad = True

        # fix the word embedding
        self.static_char_embed = nn.Embedding(self.args.static_embed_char_num, self.args.embed_char_dim)
        init.uniform(self.static_char_embed.weight, a=-np.sqrt(3 / self.args.embed_char_dim),
                     b=np.sqrt(3 / self.args.embed_char_dim))
        self.static_bichar_embed = nn.Embedding(self.args.static_embed_bichar_num, self.args.embed_bichar_dim)
        init.uniform(self.static_bichar_embed.weight, a=-np.sqrt(3 / self.args.embed_bichar_dim),
                     b=np.sqrt(3 / self.args.embed_bichar_dim))

        # load external word embedding
        if args.char_Embedding is True:
            print("char_Embedding")
            pretrained_char_weight = np.array(args.pre_char_word_vecs)
            self.static_char_embed.weight.data.copy_(torch.from_numpy(pretrained_char_weight))
            for index in range(self.args.embed_char_dim):
                self.static_char_embed.weight.data[self.args.create_static_alphabet.char_PaddingID][index] = 0
            self.static_char_embed.weight.requires_grad = False

        if args.bichar_Embedding is True:
            print("bichar_Embedding")
            pretrained_bichar_weight = np.array(args.pre_bichar_word_vecs)
            self.static_bichar_embed.weight.data.copy_(torch.from_numpy(pretrained_bichar_weight))
            # print(self.static_bichar_embed.weight.data[self.args.create_static_alphabet.bichar_PaddingID])
            # print(self.static_bichar_embed.weight.data[self.args.create_static_alphabet.bichar_UnkID])
            for index in range(self.args.embed_bichar_dim):
                self.static_bichar_embed.weight.data[self.args.create_static_alphabet.bichar_PaddingID][index] = 0
            self.static_bichar_embed.weight.requires_grad = False

        self.lstm_left = nn.LSTM(input_size=self.args.hidden_size, hidden_size=self.args.rnn_hidden_dim,
                                 dropout=self.args.dropout_lstm, bias=True)
        self.lstm_right = nn.LSTM(input_size=self.args.hidden_size, hidden_size=self.args.rnn_hidden_dim,
                                  dropout=self.args.dropout_lstm, bias=True)

        # init lstm weight and bias
        init.xavier_uniform(self.lstm_left.weight_ih_l0)
        init.xavier_uniform(self.lstm_left.weight_hh_l0)
        init.xavier_uniform(self.lstm_right.weight_ih_l0)
        init.xavier_uniform(self.lstm_right.weight_hh_l0)
        value = np.sqrt(6 / (self.args.rnn_hidden_dim + 1))
        self.lstm_left.bias_ih_l0.data.uniform_(-value, value)
        self.lstm_left.bias_hh_l0.data.uniform_(-value, value)
        self.lstm_right.bias_ih_l0.data.uniform_(-value, value)
        self.lstm_right.bias_hh_l0.data.uniform_(-value, value)

        self.hidden_l = self.init_hidden_cell(self.args.batch_size)
        self.hidden_r = self.init_hidden_cell(self.args.batch_size)

        self.dropout = nn.Dropout(self.args.dropout)
        self.dropout_embed = nn.Dropout(self.args.dropout_embed)

        self.input_dim = (self.args.embed_char_dim + self.args.embed_bichar_dim) * 2
        self.liner = nn.Linear(in_features=self.input_dim, out_features=self.args.hidden_size, bias=True)

        # init linear
        init.xavier_uniform(self.liner.weight)
        init_linear_value = np.sqrt(6 / (self.args.hidden_size + 1))
        self.liner.bias.data.uniform_(-init_linear_value, init_linear_value)

    def init_hidden_cell(self, batch_size=1):
        # the first is the hidden h
        # the second is the cell  c
        if self.args.use_cuda is True:
            return (Variable(torch.zeros(1, batch_size, self.args.rnn_hidden_dim)).cuda(),
                    Variable(torch.zeros(1, batch_size, self.args.rnn_hidden_dim)).cuda())
        else:
            return (Variable(torch.zeros(1, batch_size, self.args.rnn_hidden_dim)),
                    Variable(torch.zeros(1, batch_size, self.args.rnn_hidden_dim)))

    def init_cell_hidden(self, batch=1):
        if self.args.use_cuda is True:
            return (torch.autograd.Variable(torch.zeros(batch, self.args.rnn_hidden_dim)).cuda(),
                    torch.autograd.Variable(torch.zeros(batch, self.args.rnn_hidden_dim)).cuda())
        else:
            return (torch.autograd.Variable(torch.zeros(batch, self.args.rnn_hidden_dim)),
                    torch.autograd.Variable(torch.zeros(batch, self.args.rnn_hidden_dim)))

    def forward(self, features):
        batch_length = features.batch_length
        char_features_num = features.static_char_features.size(1)
        # fine tune
        char_features = self.char_embed(features.char_features)
        bichar_left_features = self.bichar_embed(features.bichar_left_features)
        bichar_right_features = self.bichar_embed(features.bichar_right_features)

        # fix the word embedding
        static_char_features = self.static_char_embed(features.static_char_features)
        static_bichar_l_features = self.static_bichar_embed(features.static_bichar_left_features)
        static_bichar_r_features = self.static_bichar_embed(features.static_bichar_right_features)

        # dropout
        char_features = self.dropout_embed(char_features)
        bichar_left_features = self.dropout_embed(bichar_left_features)
        bichar_right_features = self.dropout_embed(bichar_right_features)
        static_char_features = self.dropout_embed(static_char_features)
        static_bichar_l_features = self.dropout_embed(static_bichar_l_features)
        static_bichar_r_features = self.dropout_embed(static_bichar_r_features)

        # left concat
        left_concat = torch.cat((char_features, static_char_features, bichar_left_features, static_bichar_l_features), 2)
        # left_concat = left_concat.view(batch_length * char_features_num, self.input_dim)
        # right concat
        right_concat = torch.cat((char_features, static_char_features, bichar_right_features, static_bichar_r_features), 2)
        # right_concat = right_concat.view(batch_length * char_features_num, self.input_dim)

        # non-linear
        left_concat_non_linear = self.dropout(F.tanh(self.liner(left_concat)))
        # left_concat = left_concat.view(batch_length, char_features_num, self.args.rnn_hidden_dim)
        left_concat_input = left_concat_non_linear.permute(1, 0, 2)

        right_concat_non_linear = self.dropout(F.tanh(self.liner(right_concat)))
        right_concat_input = right_concat_non_linear.permute(1, 0, 2)
        # right_concat = right_concat.view(batch_length, char_features_num, self.args.rnn_hidden_dim)

        # reverse right_concat
        right_concat_input = right_concat_input.permute(1, 0, 2)
        for batch in range(batch_length):
            middle = right_concat_input.size(1) // 2
            # print(middle)
            for i, j in zip(range(0, middle, 1), range(right_concat_input.size(1) - 1, middle, -1)):
                temp = torch.zeros(right_concat_input[batch][i].data.size())
                temp.copy_(right_concat_input[batch][i].data)
                right_concat_input[batch][i].data.copy_(right_concat_input[batch][j].data)
                right_concat_input[batch][j].data.copy_(temp)
        right_concat_input = right_concat_input.permute(1, 0, 2)

        # non-linear dropout
        left_concat_input = self.dropout(left_concat_input)
        right_concat_input = self.dropout(right_concat_input)

        # init hidden cell
        self.hidden = self.init_hidden_cell(batch_size=batch_length)

        # lstm
        # lstm_left_out, _ = self.lstm_left(left_concat_input, self.hidden)
        # lstm_right_out, _ = self.lstm_right(right_concat_input, self.hidden)
        lstm_left_out, _ = self.lstm_left(left_concat_input)
        lstm_right_out, _ = self.lstm_right(right_concat_input)

        # reverse lstm_right_out
        lstm_right_out = lstm_right_out.permute(1, 0, 2)
        for batch in range(batch_length):
            middle = lstm_right_out.size(1) // 2
            for i, j in zip(range(0, middle, 1), range(lstm_right_out.size(1) - 1, middle, -1)):
                temp = torch.zeros(lstm_right_out[batch][i].size())
                temp.copy_(lstm_right_out[batch][i].data)
                lstm_right_out[batch][i].data.copy_(lstm_right_out[batch][j].data)
                lstm_right_out[batch][j].data.copy_(temp)
        lstm_right_out = lstm_right_out.permute(1, 0, 2)

        encoder_output = torch.cat((lstm_left_out, lstm_right_out), 2).permute(1, 0, 2)

        return encoder_output


