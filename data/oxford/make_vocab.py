# -*- coding: utf-8 -*-
import sys


def main(argv=None):
    with open('train.txt') as fr1, \
            open('valid.txt') as fr2, \
            open('vocab.txt', 'w') as fw:
        word_dict = {
            '<bos>': 9999999999,
            '<eos>': 9999999998,
            '<pad>': 9999999997,
            '<unk>': 9999999996
        }
        for line in fr1:
            _, _, sense = line.strip().split("\t")
            for word in sense.split(' '):
                if word not in word_dict:
                    word_dict[word] = 0
                word_dict[word] += 1

        for line in fr2:
            _, _, sense = line.strip().split("\t")
            for word in sense.split(' '):
                if word not in word_dict:
                    word_dict[word] = 0
                word_dict[word] += 1

        word_list = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)
        num = 0
        for word, freq in word_list:
            fw.write("{}\t{}\t{}\n".format(word, freq, num))
            num += 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
