# -*- coding: utf-8 -*-
import argparse


def add_model_options(parser):
    parser.add_argument('--enc_arch',
                        type=str,
                        default='cbert',
                        help='pretrained encoder arch, cbert/enbert')
    parser.add_argument('--enc_d_model',
                        type=int,
                        default=768,
                        help='dim of pretrained encoder output')
    parser.add_argument('--dec_d_model',
                        type=int,
                        default=300,
                        help='dim of decoder input')
    parser.add_argument('--dec_nhead',
                        type=int,
                        default=5,
                        help='decoder head numbers')
    parser.add_argument('--dec_nlayer',
                        type=int,
                        default=6,
                        help='decoder layer numbers')
    parser.add_argument('--dec_d_ff',
                        type=int,
                        default=2048,
                        help='dim of decoder ff layer')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--dec_activation',
                        type=str,
                        default='relu',
                        help='activation in transformer decoder, relu or gelu')
    parser.add_argument('--tie_readout',
                        action='store_true',
                        help='tie readout projection and embeddings')
    parser.add_argument('--exp_mask',
                        action='store_true',
                        help='whether to example mask')


def add_build_basic_options(parser):
    add_model_options(parser)
    parser.add_argument('--seed', type=int, default=1024, help='random seed')
    parser.add_argument('--batch_size',
                        type=int,
                        default=128,
                        help='mini batch size')
    parser.add_argument('--data_path', type=str, help='path to data dir')
    parser.add_argument('--model_dir', type=str, help='path to model dir')
    parser.add_argument('--inp_lang',
                        type=str,
                        default='zh',
                        help='input language (for xlm model)')
    parser.add_argument('--dec_max_len',
                        type=int,
                        default=30,
                        help='max length when decoding')
    parser.add_argument('--inp_max_len',
                        type=int,
                        default=100,
                        help='max length of input sequence')                       
    parser.add_argument('--sentence_bleu',
                        type=str,
                        default='./sentence-bleu',
                        help='sentence-bleu.cpp path')
    parser.add_argument('--cuda',
                        action='store_true',
                        help='whether to use cuda or not')
    parser.add_argument('--vocab_path', type=str, help='path to vocab file')
    parser.add_argument('--load_model',
                        type=str,
                        default="",
                        help='path of pretrained model')


def train_options():
    parser = argparse.ArgumentParser(
        description="Training options for X-lingual definition generation.")
    add_build_basic_options(parser)

    parser.add_argument('--max_epoch',
                        type=int,
                        default=500,
                        help='number of max epoch')
    parser.add_argument('--update_interval',
                        type=int,
                        default=1,
                        help='interval for model weights update')
    parser.add_argument('--lr',
                        type=float,
                        default=3e-4,
                        help='decoder learning rate')
    parser.add_argument('--init_lr',
                        type=float,
                        default=1e-7,
                        help='decoder init learning rate')
    parser.add_argument('--min_lr',
                        type=float,
                        default=1e-9,
                        help='decoder minimize learning rate')
    parser.add_argument('--warmup',
                        type=int,
                        default=4000,
                        help='warm up steps')
    parser.add_argument('--clip_norm',
                        type=float,
                        default=0.1,
                        help='clip norm')
    parser.add_argument('--l2', type=float, default=0, help='l2 panalty')
    parser.add_argument('--log_interval',
                        type=int,
                        default=100,
                        help='interval of print status')
    parser.add_argument('--patience',
                        type=int,
                        default=10,
                        help='early stop after n epochs of no improvement')
    parser.add_argument('--vector_path', type=str, help='path to vector file')
    parser.add_argument('--fix_embedding',
                        action='store_true',
                        help='fix pretrained embeddings for decoder')
    parser.add_argument('--fix_encoder',
                        action='store_true',
                        help='fix the pretrained language model (encoder)')
    args = parser.parse_args()
    return args


def eval_options():
    parser = argparse.ArgumentParser(
        description="Eval options for X-lingual definition generation.")
    add_build_basic_options(parser)
    parser.add_argument('--beam_size',
                        type=int,
                        default=12,
                        help='beam size of beam search') 
    args = parser.parse_args()
    return args