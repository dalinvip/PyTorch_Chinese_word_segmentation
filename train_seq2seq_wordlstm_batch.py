import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn.utils as utils
import torch.optim.lr_scheduler as lr_scheduler
import random
import hyperparams as hy
from eval import Eval
import time
torch.manual_seed(hy.seed_num)
random.seed(hy.seed_num)

"""
    train function
"""


def train(train_iter, dev_iter, test_iter, model_encoder, model_decoder, args):

    if args.Adam is True:
        print("Adam Training......")
        model_encoder_parameters = filter(lambda p: p.requires_grad, model_encoder.parameters())
        model_decoder_parameters = filter(lambda p: p.requires_grad, model_decoder.parameters())
        optimizer_encoder = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model_encoder.parameters()),
                                             lr=args.lr,
                                             weight_decay=args.init_weight_decay)
        optimizer_decoder = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model_decoder.parameters()),
                                             lr=args.lr,
                                             weight_decay=args.init_weight_decay)
        # optimizer_encoder = torch.optim.Adam(model_encoder.parameters(), lr=args.lr, weight_decay=args.init_weight_decay)
        # optimizer_decoder = torch.optim.Adam(model_decoder.parameters(), lr=args.lr, weight_decay=args.init_weight_decay)

    steps = 0
    model_count = 0
    # for dropout in train / dev / test
    model_encoder.train()
    model_decoder.train()
    time_list = []
    train_eval = Eval()
    dev_eval_seg = Eval()
    dev_eval_pos = Eval()
    test_eval_seg = Eval()
    test_eval_pos = Eval()
    for epoch in range(1, args.epochs+1):
        print("\n## 第{} 轮迭代，共计迭代 {} 次 ！##\n".format(epoch, args.epochs))
        print("optimizer_encoder now lr is {}".format(optimizer_encoder.param_groups[0].get("lr")))
        print("optimizer_decoder now lr is {} \n".format(optimizer_decoder.param_groups[0].get("lr")))

        # train time
        start_time = time.time()

        # shuffle
        random.shuffle(train_iter)
        # random.shuffle(dev_iter)
        # random.shuffle(test_iter)

        model_encoder.train()
        model_decoder.train()

        for batch_count, batch_features in enumerate(train_iter):

            model_encoder.zero_grad()
            model_decoder.zero_grad()

            maxCharSize = batch_features.char_features.size()[1]
            encoder_out = model_encoder(batch_features)
            decoder_out, state = model_decoder(batch_features, encoder_out, train=True)

            cal_train_acc(batch_features, train_eval, batch_count, decoder_out, maxCharSize, args)

            loss = torch.nn.functional.nll_loss(decoder_out, batch_features.gold_features)
            # loss = F.cross_entropy(decoder_out, batch_features.gold_features)

            loss.backward()

            if args.init_clip_max_norm is not None:
                utils.clip_grad_norm(model_encoder_parameters, max_norm=args.init_clip_max_norm)
                utils.clip_grad_norm(model_decoder_parameters, max_norm=args.init_clip_max_norm)

            optimizer_encoder.step()
            optimizer_decoder.step()

            steps += 1
            if steps % args.log_interval == 0:
                sys.stdout.write("\rbatch_count = [{}] , loss is {:.6f} , (correct/ total_num) = acc ({} / {}) = "
                                 "{:.6f}%".format(batch_count + 1, loss.data[0], train_eval.correct_num,
                                                  train_eval.gold_num, train_eval.acc() * 100))
            # if steps % args.dev_interval == 0:
            #     print("\ndev F-score")
            #     dev_eval_pos.clear()
            #     dev_eval_seg.clear()
            #     eval(dev_iter, model_encoder, model_decoder, args, dev_eval_seg, dev_eval_pos)
            #     # model_encoder.train()
            #     # model_decoder.train()
            # if steps % args.test_interval == 0:
            #     print("test F-score")
            #     test_eval_pos.clear()
            #     test_eval_seg.clear()
            #     eval(test_iter, model_encoder, model_decoder, args, test_eval_seg, test_eval_pos)
            #     print("\n")
        # train time
        end_time = time.time()
        print("\ntrain time cost: ", end_time - start_time, 's')
        time_list.append(end_time - start_time)
        if time_list is not None:
            avg_time = sum(time_list) / len(time_list)
            print("{} - {} epoch avg  time {}".format(1, epoch, avg_time))
        model_encoder.eval()
        model_decoder.eval()
        if steps is not 0:
            print("\n{} epoch dev F-score".format(epoch))
            dev_eval_pos.clear()
            dev_eval_seg.clear()
            eval(dev_iter, model_encoder, model_decoder, dev_eval_seg)
            # model_encoder.train()
            # model_decoder.train()
        if steps is not 0:
            print("{} epoch test F-score".format(epoch))
            test_eval_pos.clear()
            test_eval_seg.clear()
            eval(test_iter, model_encoder, model_decoder, test_eval_seg)
            print("\n")


def cal_train_acc(batch_features, train_eval, batch_count, decoder_out, maxCharSize, args):
    # print("calculate the acc of train ......")
    train_eval.clear()
    for id_batch in range(batch_features.batch_length):
        inst = batch_features.inst[id_batch]
        for id_char in range(inst.chars_size):
            actionID = getMaxindex(decoder_out[id_batch * maxCharSize + id_char], args)
            if actionID == inst.gold_index[id_char]:
                train_eval.correct_num += 1
        train_eval.gold_num += inst.chars_size


def jointPRF_Batch(inst, state_words, seg_eval):
    words = state_words
    count = 0
    predict_seg = []

    for idx in range(len(words)):
        w = words[idx]
        predict_seg.append('[' + str(count) + ',' + str(count + len(w)) + ']')
        count += len(w)

    seg_eval.gold_num += len(inst.gold_seg)
    seg_eval.predict_num += len(predict_seg)
    for p in predict_seg:
        if p in inst.gold_seg:
            seg_eval.correct_num += 1


def getMaxindex(decode_out_acc, args):
    # print("get max index ......")
    max = decode_out_acc.data[0]
    maxIndex = 0
    for idx in range(1, args.label_size):
        if decode_out_acc.data[idx] > max:
            max = decode_out_acc.data[idx]
            maxIndex = idx
    return maxIndex


def eval(data_iter, model_encoder, model_decoder, eval_seg):
    # eval time
    eval_start = time.time()
    for batch_features in data_iter:
        encoder_out = model_encoder(batch_features)
        decoder_out, state = model_decoder(batch_features, encoder_out, train=False)
        for i in range(batch_features.batch_length):
            jointPRF_Batch(batch_features.inst[i], state.words[i], eval_seg)

    # calculate the time
    print("eval time cost: ", time.time() - eval_start, 's')

    # calculate the F-Score
    p, r, f = eval_seg.getFscore()
    print("seg: precision = {}%  recall = {}% , f-score = {}%\n".format(p, r, f))




