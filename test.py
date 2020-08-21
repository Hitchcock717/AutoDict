# -*- coding: utf-8 -*-
import sys
import os
import math
import glob
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np

from src import dataset, model, utils, options
import evaluate


def load_checkpoint(model_with_loss, path):
    checkpoint = torch.load(path)
    if torch.cuda.device_count() > 1:
        model_with_loss.module.model.load_state_dict(checkpoint)
    else:
        model_with_loss.model.load_state_dict(checkpoint)


def test(args, test_iter, vocab, path=None):
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
                            use_pretrained_encoder=True,
                            exp_mask=args.exp_mask)
    criterion_ce = nn.NLLLoss(ignore_index=vocab.stoi['<pad>'],
                              reduction='mean')
    criterion = utils.LabelSmoothKLDiv(args.len_vocab,
                                       ignore_idx=vocab.stoi['<pad>'],
                                       reduction='batchmean')

    model_with_loss = model.ModelWithLoss(mymodel,
                                          criterion_ce,
                                          criterion)
    if args.cuda:
        model_with_loss = model_with_loss.cuda()

    if torch.cuda.device_count() > 1:
        model_with_loss = nn.DataParallel(model_with_loss)

    if path:
        load_checkpoint(model_with_loss, path)
    else:
        path = glob.glob(os.path.join(args.model_dir, 'model-best-*.pt'))[0]
        load_checkpoint(model_with_loss, path)

    test_loss_ce, test_loss = evaluate.eval_loss(
        test_iter, model_with_loss, args.cuda)
    test_hyp_greedy = evaluate.greedy(test_iter,
                                        vocab,
                                        model_with_loss,
                                        args.cuda,
                                        max_len=args.dec_max_len)
    test_hyp_bs = evaluate.beam_search(test_iter,
                                        vocab,
                                        model_with_loss,
                                        args.cuda,
                                        args.beam_size,
                                        args.dec_max_len)

    test_bleu_greedy = evaluate.bleu(test_hyp_greedy,
                                    test_iter,
                                    bleu_path=args.sentence_bleu,
                                    nltk='sentence')
    test_bleu_bs = evaluate.bleu(test_hyp_bs,
                                    test_iter,
                                    bleu_path=args.sentence_bleu,
                                    nltk='sentence')

    hyp_write_path_greedy = os.path.join(args.model_dir, 'hyp_greedy.txt')
    hyp_write_path_bs = os.path.join(args.model_dir, 'hyp_beam_search.txt')
    with open(hyp_write_path_greedy, 'w') as fw1, open(hyp_write_path_bs, 'w') as fw2:
        for idx, sense in enumerate(test_hyp_greedy):
            fw1.write("{}\t{}\n".format(test_iter.dataset.raw_data[idx][0], ' '.join(sense)))
        for idx, sense in enumerate(test_hyp_bs):
            fw2.write("{}\t{}\n".format(test_iter.dataset.raw_data[idx][0], ' '.join(sense)))
    print('=' * 80)
    print('| End of Decoding | Loss {:.2f} | CELoss {:.2f} '
        '| PPL {:.2f} | Test BLEU (Greedy/Beam search) {:.2f}/{:.2f}'.format(
            test_loss, test_loss_ce, math.exp(test_loss_ce),
            test_bleu_greedy * 100, test_bleu_bs * 100))
    print('=' * 80)
    

def main(args):
    torch.set_num_threads(4)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    print("Loading Datasets ...")
    test_data = dataset.obtain_dataset(args.data_path, args.enc_arch,
                                       args.inp_lang, args.inp_max_len, 'test')
    test_iter = data.DataLoader(test_data,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=4,
                                pin_memory=True)
    vocab = utils.Vocab(args.vocab_path)
    args.len_vocab = len(vocab.itos)

    print(args)

    assert args.load_model, "--load_model must be known"
    if test(args, test_iter, vocab, path=args.load_model):
        print("Inference Done.")


if __name__ == '__main__':
    args = options.eval_options()
    sys.exit(main(args))
