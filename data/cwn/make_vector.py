import numpy as np


pretrained_file = './merge_sgns_bigram_char300.txt'
vocab_size = 14905
with open('./vocab.txt') as fr1, open(pretrained_file) as fr2:
    vocab = []
    for line in fr1:
        vocab.append(line.strip().split('\t')[0])
    word2vec = {}
    vec_dim = 0
    for line in fr2:
        content = line.strip().split(' ')
        if len(content) == 2:
            continue
        word = content[0]
        vec = ' '.join(content[1:])
        vec = np.fromstring(vec, sep=' ')
        word2vec[word] = vec
        if not vec_dim:
            vec_dim = len(vec)

    matrix = np.zeros((len(vocab), vec_dim))
    for idx, word in enumerate(vocab):
        if word in word2vec:
            vec = word2vec[word]
        else:
            vec = np.random.uniform(-0.05, 0.05, (vec_dim, ))
        matrix[idx] = vec
    arr = np.array(matrix)
    np.save("./vec.cwn.npy", arr)
    print("save .npy done")
