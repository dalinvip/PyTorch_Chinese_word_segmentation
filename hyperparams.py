import torch
import random
torch.manual_seed(121)
random.seed(121)
# random seed
seed_num = 233


class Hyperparams():
    def __init__(self):

        # data path
        self.train_path = "./data_test/TrainingCorpus_16.seg.txt"
        self.dev_path = "./data_test/devCorpus_16.seg.txt"
        self.test_path = "./data_test/TestCorpus_gold_16.seg.txt"

        # self.train_path = "./data/TrainingCorpus_16.seg.txt"
        # self.dev_path = "./data/devCorpus_16.seg.txt"
        # self.test_path = "./data/TestCorpus_gold_16.seg.txt"

        self.learning_rate = 0.001
        # self.learning_rate_decay = 0.9   # value is 1 means not change lr
        self.learning_rate_decay = 1   # value is 1 means not change lr
        self.epochs = 200
        self.train_batch_size = 16
        self.dev_batch_size = 16
        self.test_batch_size = 16
        self.log_interval = 1
        self.dev_interval = 4000000
        self.test_interval = 4000000
        self.save_dir = "snapshot"
        self.shuffle = False
        self.epochs_shuffle = True
        self.dropout = 0.25
        self.dropout_embed = 0.25
        self.dropout_lstm = 0.5
        self.max_norm = None
        self.clip_max_norm = 10
        self.static = False
        # model
        self.Wordlstm = True
        self.Encoder_LSTM = False
        self.Encoder_LSTMCell = True
        # select optim algorhtim to train
        self.Adam = True
        self.SGD = False
        self.Adadelta = False
        self.optim_momentum_value = 0.9
        # min freq to include during built the vocab, default is 1
        self.min_freq = 1
        self.word_min_freq = 1
        self.char_min_freq = 2
        self.bichar_min_freq = 2
        # word_Embedding
        self.word_Embedding = True
        self.word_Embedding_Path = "./word2vec/glove.sentiment.conj.pretrained.txt"
        self.fix_Embedding = False
        self.embed_dim = 300
        self.embed_char_dim = 200
        self.embed_bichar_dim = 200
        # word_Embedding_Path = "./word2vec/glove.840B.300d.handled.Twitter.txt"
        self.char_Embedding = False
        self.char_Embedding_path = "./word_embedding/char.vec"
        self.bichar_Embedding = False
        # self.bichar_Embedding_Path = "./word_embedding/char.vec"
        self.bichar_Embedding_Path = "./word_embedding/bichar-small.vec"
        # self.bichar_Embedding_Path = "./word_embedding/convert_bichar.txt"
        # self.pos_num = None
        self.pos_dim = 100

        self.rnn_hidden_dim = 200
        self.hidden_size = 200
        self.rnn_num_layers = 1
        self.gpu_device = 0
        self.use_cuda = False
        self.snapshot = None
        self.num_threads = 1
        # whether to init w
        self.init_weight = True
        self.init_weight_value = 6.0
        # L2 weight_decay
        self.weight_decay = 1e-8  # default value is zero in Adam SGD
        # self.weight_decay = 0   # default value is zero in Adam SGD
        # whether to delete the model after test acc so that to save space
        self.rm_model = True



