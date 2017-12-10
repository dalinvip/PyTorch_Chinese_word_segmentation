#! /usr/bin/env python
import os
import argparse
import datetime
import torch
from loaddata.Load_external_word_embedding import Word_Embedding
from loaddata.Dataloader import load_data, instance
from loaddata.Alphabet import Create_Alphabet, Alphabet
from loaddata.Batch_Iterator import Iterators
from models import decoder_wordlstm_batch
from models import encoder_wordlstm
from models import encoder_wordlstmcell
import train_seq2seq_wordlstm_batch
import multiprocessing as mu
import shutil
import random
import hyperparams
import hyperparams as hy
import time
# solve encoding
from imp import reload
import sys
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)

# init hyperparams instance
hyperparams = hyperparams.Hyperparams()

# random seed
torch.manual_seed(hy.seed_num)
random.seed(hy.seed_num)

parser = argparse.ArgumentParser(description="Chinese word segmentation")
# learning
parser.add_argument('-lr', type=float, default=hyperparams.learning_rate, help='initial learning rate [default: 0.001]')
parser.add_argument('-learning_rate_decay', type=float, default=hyperparams.learning_rate_decay, help='initial learning_rate_decay rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=hyperparams.epochs, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=hyperparams.train_batch_size, help='batch size for training [default: 64]')
parser.add_argument('-dev-batch-size', type=int, default=hyperparams.dev_batch_size, help='batch size for training [default: 64]')
parser.add_argument('-test-batch-size', type=int, default=hyperparams.test_batch_size, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=hyperparams.log_interval,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-dev-interval', type=int, default=hyperparams.dev_interval, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-test-interval', type=int, default=hyperparams.test_interval, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default=hyperparams.save_dir, help='where to save the snapshot')
# data path
parser.add_argument('-train_path', type=str, default=hyperparams.train_path, help='train data path')
parser.add_argument('-dev_path', type=str, default=hyperparams.dev_path, help='dev data path')
parser.add_argument('-test_path', type=str, default=hyperparams.test_path, help='test data path')
# shuffle data
parser.add_argument('-shuffle', action='store_true', default=hyperparams.shuffle, help='shuffle the data when load data' )
parser.add_argument('-epochs_shuffle', action='store_true', default=hyperparams.epochs_shuffle, help='shuffle the data every epoch' )
# optim select
parser.add_argument('-Adam', action='store_true', default=hyperparams.Adam, help='whether to select Adam to train')
parser.add_argument('-SGD', action='store_true', default=hyperparams.SGD, help='whether to select SGD to train')
parser.add_argument('-Adadelta', action='store_true', default=hyperparams.Adadelta, help='whether to select Adadelta to train')
parser.add_argument('-momentum_value', type=float, default=hyperparams.optim_momentum_value, help='value of momentum in SGD')
# model
parser.add_argument('-rm_model', action='store_true', default=hyperparams.rm_model, help='whether to delete the model after test acc so that to save space')
parser.add_argument('-init_weight', action='store_true', default=hyperparams.init_weight, help='init w')
parser.add_argument('-init_weight_value', type=float, default=hyperparams.init_weight_value, help='value of init w')
parser.add_argument('-init_weight_decay', type=float, default=hyperparams.weight_decay, help='value of init L2 weight_decay')
parser.add_argument('-init_clip_max_norm', type=float, default=hyperparams.clip_max_norm, help='value of init clip_max_norm')
parser.add_argument('-dropout', type=float, default=hyperparams.dropout, help='the probability for dropout [default: 0.5]')
parser.add_argument('-dropout_embed', type=float, default=hyperparams.dropout_embed, help='the probability for dropout [default: 0.5]')
parser.add_argument('-dropout_lstm', type=float, default=hyperparams.dropout_lstm, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=hyperparams.max_norm, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=hyperparams.embed_dim, help='number of embedding dimension [default: 128]')
parser.add_argument('-static', action='store_true', default=hyperparams.static, help='fix the embedding')
parser.add_argument('-Wordlstm', action='store_true', default=hyperparams.Wordlstm, help='whether to use Wordlstm decoder model')
parser.add_argument('-Encoder_LSTM', action='store_true', default=hyperparams.Encoder_LSTM, help='whether to use BiLSTM_1 model')
parser.add_argument('-Encoder_LSTMCell', action='store_true', default=hyperparams.Encoder_LSTMCell, help='whether to use LSTM model')
parser.add_argument('-fix_Embedding', action='store_true', default=hyperparams.fix_Embedding, help='whether to fix word embedding during training')
parser.add_argument('-word_Embedding', action='store_true', default=hyperparams.word_Embedding, help='whether to load word embedding')
parser.add_argument('-word_Embedding_Path', type=str, default=hyperparams.word_Embedding_Path, help='filename of model snapshot [default: None]')
parser.add_argument('-rnn_hidden_dim', type=int, default=hyperparams.rnn_hidden_dim, help='the number of embedding dimension in LSTM hidden layer')
parser.add_argument('-rnn_num_layers', type=int, default=hyperparams.rnn_num_layers, help='the number of embedding dimension in LSTM hidden layer')
parser.add_argument('-min_freq', type=int, default=hyperparams.min_freq, help='min freq to include during built the vocab')
parser.add_argument('-word_min_freq', type=int, default=hyperparams.word_min_freq, help='word_min_freq')
parser.add_argument('-char_min_freq', type=int, default=hyperparams.char_min_freq, help='char_min_freq')
parser.add_argument('-bichar_min_freq', type=int, default=hyperparams.bichar_min_freq, help='bichar_min_freq')
# encoder model and decoder model
parser.add_argument('-embed_char_dim', type=int, default=hyperparams.embed_char_dim, help='number of char embedding dimension [default: 200]')
parser.add_argument('-embed_bichar_dim', type=int, default=hyperparams.embed_bichar_dim, help='number of bichar embedding dimension [default: 200]')
parser.add_argument('-hidden_size', type=int, default=hyperparams.hidden_size, help='hidden dimension [default: 200]')
# word embedding
parser.add_argument('-pos_dim', type=int, default=hyperparams.pos_dim, help='pos_dim')
parser.add_argument('-char_Embedding', action='store_true', default=hyperparams.char_Embedding, help='whether to load char embedding')
parser.add_argument('-char_Embedding_path', type=str, default=hyperparams.char_Embedding_path, help='char_Embedding_path')
parser.add_argument('-bichar_Embedding', action='store_true', default=hyperparams.bichar_Embedding, help='whether to load bichar embedding')
parser.add_argument('-bichar_Embedding_Path', type=str, default=hyperparams.bichar_Embedding_Path, help='bichar_Embedding_Path')
# seed number
parser.add_argument('-seed_num', type=float, default=hy.seed_num, help='value of init seed number')
# nums of threads
parser.add_argument('-num_threads', type=int, default=hyperparams.num_threads, help='the num of threads')
# device
parser.add_argument('-gpu_device', type=int, default=hyperparams.gpu_device, help='gpu number that usr in training')
parser.add_argument('-use_cuda', action='store_true', default=hyperparams.use_cuda, help='disable the gpu')
# option
args = parser.parse_args()


# load data / create alphabet / create iterator
def dalaloader(args):
    print("loading data......")
    # read file
    data_loader = load_data()
    train_data, dev_data, test_data = data_loader.load_data(path=[args.train_path, args.dev_path, args.test_path],
                                                            shuffle=args.shuffle)
    print(train_data)
    # create the alphabet
    create_alphabet = Create_Alphabet(min_freq=args.min_freq, word_min_freq=args.word_min_freq,
                                      char_min_freq=args.char_min_freq, bichar_min_freq=args.bichar_min_freq)
    create_alphabet.createAlphabet(train_data=train_data)

    create_static_alphabet = Create_Alphabet(min_freq=args.min_freq, word_min_freq=args.min_freq,
                                             char_min_freq=args.min_freq, bichar_min_freq=args.min_freq)
    create_static_alphabet.createAlphabet(train_data=train_data, dev_data=dev_data, test_data=test_data)
    # create iterator
    create_iter = Iterators()
    train_iter, dev_iter, test_iter = create_iter.createIterator(batch_size=[args.batch_size, args.dev_batch_size,
                                                                             args.test_batch_size],
                                                                 data=[train_data, dev_data, test_data],
                                                                 operator=create_alphabet,
                                                                 operator_static=create_static_alphabet, args=args)
    return train_iter, dev_iter, test_iter, create_alphabet, create_static_alphabet


# get iter
train_iter, dev_iter, test_iter, create_alphabet, create_static_alphabet = dalaloader(args)


if args.char_Embedding is True:
    print("loading char embedding.......")
    char_word_vecs_dict = Word_Embedding().load_my_vecs(path=args.char_Embedding_path,
                                                        vocab=create_static_alphabet.char_alphabet.id2words,
                                                        freqs=None, k=args.embed_char_dim)
    print("avg handle tha oov words")
    char_word_vecs = Word_Embedding().add_unknown_words_by_avg(word_vecs=char_word_vecs_dict,
                                                               vocab=create_static_alphabet.char_alphabet.id2words,
                                                               k=args.embed_char_dim)
    # char_word_vecs = Word_Embedding().add_unknown_words_by_uniform(word_vecs=char_word_vecs,
    #                                                                vocab=create_static_alphabet.char_alphabet.id2words,
    #                                                                k=args.embed_char_dim)
    # print(char_word_vecs)

if args.bichar_Embedding is True:
    print("loading bichar embedding.......")
    bichar_word_vecs_dict = Word_Embedding().load_my_vecs(path=args.bichar_Embedding_Path,
                                                          vocab=create_static_alphabet.bichar_alphabet.id2words,
                                                          freqs=None, k=args.embed_bichar_dim)
    print("avg handle tha oov words")
    bichar_word_vecs = Word_Embedding().add_unknown_words_by_avg(word_vecs=bichar_word_vecs_dict,
                                                                 vocab=create_static_alphabet.bichar_alphabet.id2words,
                                                                 k=args.embed_bichar_dim)
    # bichar_word_vecs = Word_Embedding().add_unknown_words_by_uniform(word_vecs=bichar_word_vecs,
    #                                                                  vocab=create_static_alphabet.bichar_alphabet.id2words,
    #                                                                  k=args.embed_bichar_dim)
    # print(bichar_word_vecs)

# # handle external word embedding to file for convenience
# from loaddata.handle_wordEmbedding2File import WordEmbedding2File
# wordembedding = WordEmbedding2File(wordEmbedding_path=args.bichar_Embedding_Path,
#                                    vocab=create_static_alphabet.bichar_alphabet.id2words, k_dim=200)
# wordembedding.handle()

# update parameters
if args.char_Embedding is True:
    args.pre_char_word_vecs = char_word_vecs
if args.bichar_Embedding is True:
    args.pre_bichar_word_vecs = bichar_word_vecs

args.use_cuda = (args.use_cuda) and torch.cuda.is_available()

args.embed_char_num = create_alphabet.char_alphabet.m_size
args.embed_bichar_num = create_alphabet.bichar_alphabet.m_size
args.static_embed_char_num = create_static_alphabet.char_alphabet.m_size
args.static_embed_bichar_num = create_static_alphabet.bichar_alphabet.m_size
print("create_alphabet.char_alphabet.m_size", create_alphabet.char_alphabet.m_size)
print("create_static_alphabet.char_alphabet.m_size", create_static_alphabet.char_alphabet.m_size)


args.label_size = create_alphabet.label_alphabet.m_size
args.pos_size = create_alphabet.pos_alphabet.m_size

args.create_alphabet = create_alphabet
args.create_static_alphabet = create_static_alphabet
# save file
mulu = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
args.mulu = mulu
args.save_dir = os.path.join(args.save_dir, mulu)
if not os.path.isdir(args.save_dir):
    os.makedirs(args.save_dir)


# print parameters
print("\nParameters:")
if os.path.exists("./Parameters.txt"):
    os.remove("./Parameters.txt")
file = open("Parameters.txt", "a")
for attr, value in sorted(args.__dict__.items()):
    # if attr.upper() != "PRE_CHAR_WORD_VECS" and attr.upper() != "PRE_BICHAR_WORD_VECS":
    #     print("\t{}={}".format(attr.upper(), value))
    if attr.upper() == "PRE_CHAR_WORD_VECS" or attr.upper() == "PRE_BICHAR_WORD_VECS":
        continue
    print("\t{}={}".format(attr.upper(), value))
    file.write("\t{}={}\n".format(attr.upper(), value))
file.close()
shutil.copy("./Parameters.txt", "./snapshot/" + mulu + "/Parameters.txt")
shutil.copy("./hyperparams.py", "./snapshot/" + mulu)


# load model
model_encoder = None
model_decoder = None
# wordlstm and batch
if args.Wordlstm is True:
    print("loading word lstm decoder model")
    model_decoder = decoder_wordlstm_batch.Decoder_WordLstm(args=args)
    if args.Encoder_LSTM is True:
        model_encoder = encoder_wordlstm.Encoder_WordLstm(args)
    elif args.Encoder_LSTMCell is True:
        model_encoder = encoder_wordlstmcell.Encoder_WordLstm(args)
else:
    print("please choose Wordlstm is True......")
print(model_encoder)
print(model_decoder)
if args.use_cuda is True:
    print("using cuda......")
    model_encoder = model_encoder.cuda()
    model_decoder = model_decoder.cuda()

# train
print("\n CPU Count is {} and Current Process is {} \n".format(mu.cpu_count(), mu.current_process()))
# set thread number
torch.set_num_threads(args.num_threads)
print("train_seq2seq_wordlstm")
train_seq2seq_wordlstm_batch.train(train_iter=train_iter, dev_iter=dev_iter, test_iter=test_iter,
                                   model_encoder=model_encoder,
                                   model_decoder=model_decoder, args=args)


