# -*- coding: utf-8 -*-
import sys
import os
import time
import math
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils import data

from src import dataset, model, utils, options
import evaluate


def save_checkpoint(model_with_loss, model_dir, epoch):
    del_path = glob.glob(os.path.join(model_dir, 'model-best-*.pt'))
    for d in del_path:
        os.remove(d)
    save_path = os.path.join(model_dir, 'model-best-{:03d}.pt'.format(epoch))
    if torch.cuda.device_count() > 1:
        torch.save(model_with_loss.module.model.state_dict(), save_path)
    else:
        torch.save(model_with_loss.model.state_dict(), save_path)


def load_checkpoint(model_with_loss, path):
    checkpoint = torch.load(path)
    if torch.cuda.device_count() > 1:
        model_with_loss.module.model.load_state_dict(checkpoint)
    else:
        model_with_loss.model.load_state_dict(checkpoint)


def is_no_improve(val_loss, best_val_loss, val_bleu_sentence, best_bleu,
                  epoch):
    if val_loss < best_val_loss:
        save = True
        best_val_loss = val_loss
        best_bleu = val_bleu_sentence
        no_improv = 0
    elif val_bleu_sentence > best_bleu:
        save = True
        best_bleu = val_bleu_sentence
        no_improv = 0
    else:
        save = False
        no_improv = 1
    return no_improv, best_val_loss, best_bleu, save


def train(args, train_iter, valid_iter, vocab):
    mymodel = model.MyModel(args.enc_arch,
                            args.enc_d_model,
                            args.len_vocab,
                            args.dec_d_model,
                            args.dec_nhead,
                            args.dec_nlayer,
                            args.tie_readout,
                            args.dec_d_ff,
                            args.dropout,
                            args.dec_activation,
                            vector_path=args.vector_path,
                            fix_embedding=args.fix_embedding,
                            use_pretrained_encoder=True,
                            fix_encoder=args.fix_encoder,
                            exp_mask=args.exp_mask)
    criterion_ce = nn.NLLLoss(ignore_index=vocab.stoi['<pad>'],
                              reduction='mean')
    criterion = utils.LabelSmoothKLDiv(args.len_vocab,
                                       ignore_idx=vocab.stoi['<pad>'],
                                       reduction='batchmean')

    model_with_loss = model.ModelWithLoss(mymodel,
                                          criterion_ce,
                                          criterion)

    base_optim = optim.Adam(filter(lambda p: p.requires_grad,
                                   model_with_loss.parameters()),
                            betas=(0.9, 0.98),
                            eps=1e-9,
                            weight_decay=args.l2)
    advanced_optim = utils.InverseSqureRootOptim(base_optim, args.init_lr,
                                                 args.lr, args.min_lr,
                                                 args.warmup)
    if args.cuda:
        model_with_loss = model_with_loss.cuda()

    if torch.cuda.device_count() > 1:
        model_with_loss = nn.DataParallel(model_with_loss)

    if args.load_model:
        load_checkpoint(model_with_loss, args.load_model)

    batch_num = len(train_iter)
    best_val_loss = 9999999
    best_bleu = -1
    no_improvement = 0
    for epoch in range(1, args.max_epoch + 1):
        total_loss_ce = 0
        total_loss = 0
        start_time = time.time()
        epoch_start_time = time.time()
        for i, item in enumerate(train_iter):
            model_with_loss.train()
            if args.cuda:
                item = utils.to_cuda(item)
            _, loss_ce, loss = model_with_loss(item)
            loss_ce = loss_ce.mean()
            loss = loss.mean()
            total_loss_ce += loss_ce.item()
            total_loss += loss.item()
            loss_ce = loss_ce / args.update_interval
            loss = loss / args.update_interval
            loss.backward()

            if (i + 1) % args.update_interval == 0 or (i + 1) == batch_num:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad,
                           model_with_loss.parameters()), args.clip_norm)
                advanced_optim.step()
                model_with_loss.zero_grad()

            if (i + 1) % args.log_interval == 0:
                elapsed = time.time() - start_time
                cur_loss_ce = total_loss_ce / args.log_interval
                cur_loss = total_loss / args.log_interval
                print("| Epoch {:3d} | {:d}/{:d} batches | {:.2f} ms/batch "
                      "| LR {:.9f} | Step {:d} | Loss {:.2f} | CELoss {:.2f} "
                      "| PPL {:.2f}".format(epoch, i + 1, batch_num,
                                            elapsed * 1000 / args.log_interval,
                                            advanced_optim._rate,
                                            advanced_optim._step,
                                            cur_loss, cur_loss_ce,
                                            math.exp(cur_loss_ce)))
                total_loss_ce = 0
                total_loss = 0
                start_time = time.time()
        val_loss_ce, val_loss = evaluate.eval_loss(
            valid_iter, model_with_loss, args.cuda)
        val_hyp = evaluate.greedy(valid_iter,
                                           vocab,
                                           model_with_loss,
                                           args.cuda,
                                           max_len=args.dec_max_len)
        val_bleu_corpus = evaluate.bleu(val_hyp,
                                        valid_iter,
                                        bleu_path=args.sentence_bleu,
                                        nltk='corpus')
        val_bleu_sentence = evaluate.bleu(val_hyp,
                                          valid_iter,
                                          bleu_path=args.sentence_bleu,
                                          nltk='sentence')

        ret, best_val_loss, best_bleu, save = is_no_improve(
            val_loss_ce, best_val_loss, val_bleu_sentence, best_bleu, epoch)
        no_improvement = no_improvement + ret if ret else 0
        if save:
            save_checkpoint(model_with_loss, args.model_dir, epoch)

        print('-' * 80)
        print("| End of epoch {:3d} | Time {:.2f}s | LR {:.9f} | Step {:d}".
              format(epoch,
                     time.time() - epoch_start_time, advanced_optim._rate,
                     advanced_optim._step))
        print("| Loss {:.2f} | CELoss {:.2f} | PPL {:.2f} | BLEU (C/S) "
              "{:.2f}/{:.2f}".format(val_loss, val_loss_ce,
                                     math.exp(val_loss_ce),
                                     val_bleu_corpus * 100,
                                     val_bleu_sentence * 100))
        print("| Not Improved {:d}".format(no_improvement))
        print('-' * 80)

        if no_improvement >= args.patience:
            break


def main(args):
    torch.set_num_threads(4)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    print("Loading Datasets ...")
    train_data = dataset.obtain_dataset(args.data_path, args.enc_arch,
                                        args.inp_lang, args.inp_max_len, 'train')
    valid_data = dataset.obtain_dataset(args.data_path, args.enc_arch,
                                        args.inp_lang, args.inp_max_len, 'valid')
    train_iter = data.DataLoader(train_data,
                                 batch_size=args.batch_size,
                                 shuffle=True,
                                 num_workers=4,
                                 pin_memory=True)
    valid_iter = data.DataLoader(valid_data,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 pin_memory=True)
    vocab = utils.Vocab(args.vocab_path)
    args.len_vocab = len(vocab.itos)

    print(args)

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if train(args, train_iter, valid_iter, vocab):
        print("End of the training stage.")


if __name__ == '__main__':
    args = options.train_options()
    sys.exit(main(args))
