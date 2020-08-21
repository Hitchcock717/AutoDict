# -*- coding: utf-8 -*-
import os
import json
import pickle
import numpy as np
from torch.utils import data

from src.utils import obtain_tokenizer


class MyDataset(data.Dataset):
    def __init__(self, data_path, enc_arch, inp_lang, inp_max_len, mode='train'):
        self.mode = mode
        self.data, self.raw_data = self.read_data(data_path, enc_arch,
                                                  inp_lang, inp_max_len)

    def read_data(self, data_path, enc_arch, inp_lang, inp_max_len):
        out_i2s = []
        with open(os.path.join(data_path, 'vocab.txt')) as fr:
            for line in fr:
                out_i2s.append(line.strip().split('\t')[0])
        out_s2i = {s: i for i, s in enumerate(out_i2s)}

        inp_max_length = inp_max_len
        out_max_length = 55
        inp_position_ids = np.arange(inp_max_length, dtype=int)

        tokenizer = obtain_tokenizer(enc_arch)
        if getattr(tokenizer, 'lang2id', 0):
            lang_ids = [tokenizer.lang2id[inp_lang]] * inp_max_length
        else:
            lang_ids = [-1] * inp_max_length

        raw_data = []
        data_list = []
        with open(os.path.join(data_path, self.mode + '.txt')) as fr:
            for line in fr:
                word, exp, sense = line.strip().split('\t')
                raw_data.append((word, exp, sense))
                inp = tokenizer.encode_plus(word,
                                            exp,
                                            max_length=inp_max_length,
                                            pad_to_max_length=True)
                if not inp.get('token_type_ids', 0):
                    inp['token_type_ids'] = [0] * inp_max_length
                inp['position_ids'] = inp_position_ids
                inp['lang_ids'] = lang_ids
                inp['exp_mask'] = [1] * inp_max_length
                array_inp = {
                    k: np.asarray(v, dtype=int)
                    for k, v in inp.items()
                }
                array_inp['exp_mask'][len(tokenizer.encode(word)):] = 0
                out = self.get_out_seq(sense, out_s2i, out_max_length)
                array_out = {
                    k: np.asarray(v, dtype=int)
                    for k, v in out.items()
                }
                data_list.append((array_inp, array_out))
        return data_list, raw_data

    def get_out_seq(self, sense, out_s2i, out_max_length):
        output_ids = [out_s2i['<bos>']]
        attention_mask = [1]
        for word in sense.split(' '):
            if word in out_s2i:
                output_ids.append(out_s2i[word])
            else:
                output_ids.append(out_s2i['<unk>'])
            attention_mask.append(1)
        output_ids.append(out_s2i['<eos>'])
        attention_mask.append(1)

        for i in range(out_max_length - len(sense.split(' ')) - 2):
            output_ids.append(out_s2i['<pad>'])
            attention_mask.append(0)

        out = {
            'output_ids_x': output_ids[:-1],
            'output_ids_y': output_ids[1:],
            'attention_mask': attention_mask[:-1],
        }
        return out

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ret = {
            'input': {
                'input_ids': self.data[idx][0]['input_ids'],
                'token_type_ids': self.data[idx][0]['token_type_ids'],
                'attention_mask': self.data[idx][0]['attention_mask'],
                'lang_ids': self.data[idx][0]['lang_ids'],
                'position_ids': self.data[idx][0]['position_ids'],
                'exp_mask': self.data[idx][0]['exp_mask']
            },
            'output': {
                'output_ids_x': self.data[idx][1]['output_ids_x'],
                'output_ids_y': self.data[idx][1]['output_ids_y'],
                'attention_mask': self.data[idx][1]['attention_mask']
            }
        }
        return ret


def obtain_dataset(data_path, enc_arch, inp_lang, inp_max_len, mode='train'):
    assert mode in ['train', 'valid', 'test'], \
        "mode must be train/valid/test"
    bin_path = os.path.join(data_path, 'bin')
    if not os.path.exists(bin_path):
        os.makedirs(bin_path)
    dataset_fname = "{}.{}.pkl".format(enc_arch, mode)
    dataset_path = os.path.join(bin_path, dataset_fname)
    if os.path.exists(dataset_path):
        with open(dataset_path, 'rb') as fr:
            dataset = pickle.load(fr)
    else:
        dataset = MyDataset(data_path, enc_arch, inp_lang, inp_max_len, mode)
        with open(dataset_path, 'wb') as fw:
            pickle.dump(dataset, fw)
    return dataset
