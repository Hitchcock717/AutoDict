# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import random
import string
from subprocess import Popen, PIPE
from nltk.translate import bleu_score
from collections import defaultdict
from sklearn import metrics
from src.beamsearch import BeamBatch
from src import utils

sample_results_id = np.random.randint(0, 1000, (5, ))


def eval_loss(data_iter, model_with_loss, cuda):
    total_loss_ce = 0
    total_loss = 0
    acc_list = []
    p_list = []
    r_list = []
    f1_list = []
    model_with_loss.eval()
    with torch.no_grad():
        for i, item in enumerate(data_iter):
            if cuda:
                item = utils.to_cuda(item)
            output, loss_ce, loss = model_with_loss(item)
            total_loss_ce += loss_ce.mean().item()
            total_loss += loss.mean().item()
    mean_loss_ce = total_loss_ce / len(data_iter)
    mean_loss = total_loss / len(data_iter)
    return mean_loss_ce, mean_loss


def beam_search(data_iter,
                vocab,
                model_with_loss,
                cuda,
                beam_size,
                max_len=100,
                pos_vocab=None):
    model_with_loss.eval()
    with torch.no_grad():
        results = []
        all_scores = []
        for i, item in enumerate(data_iter):
            if cuda:
                item = utils.to_cuda(item)
            ori_memory = model_with_loss(item, mode='encode') # 32,100,768
            beam_batch = BeamBatch(ori_memory.size(0), beam_size,
                                   vocab.stoi['<bos>'], vocab.stoi['<eos>'],
                                   cuda)

            ori_inp_mask = item['input']['attention_mask']
            item['output']['output_ids_x'] = beam_batch.get_gen_seq()
            item['output']['attention_mask'] = torch.ones_like(
                item['output']['output_ids_x'])
            item['input']['attention_mask'] = beam_batch.expand_beams(
                ori_inp_mask)
            memory = beam_batch.expand_beams(ori_memory) # 384,100,768

            for j in range(max_len):
                output = model_with_loss(item, mode='decode', memory=memory) #384,1,14950 -- 1,14950
                next_output_feed, done = beam_batch.step(output[:, -1])
                if done:
                    break
                item['output']['output_ids_x'] = next_output_feed
                item['output']['attention_mask'] = torch.ones_like(
                    next_output_feed)
                item['input']['attention_mask'] = beam_batch.expand_beams(
                    ori_inp_mask)
                memory = beam_batch.expand_beams(ori_memory)

            generated_seq, scores = beam_batch.get_topn(1)
            for seq_id in generated_seq:
                seq = []
                for word_id in seq_id:
                    if word_id == vocab.stoi['<bos>']:
                        continue
                    if word_id == vocab.stoi['<eos>']:
                        break
                    seq.append(vocab.itos[word_id])
                # seq = ' '.join(seq)
                results.append(seq)
    return results


def greedy(data_iter,
           vocab,
           model_with_loss,
           cuda,
           max_len=100):
    model_with_loss.eval()
    with torch.no_grad():
        results = []
        for i, item in enumerate(data_iter):
            if cuda:
                item = utils.to_cuda(item)
            memory = model_with_loss(item, mode='encode')

            item['output']['output_ids_x'] = item['output'][
                'output_ids_x'][:, 0].unsqueeze(1)
            item['output']['attention_mask'] = item['output'][
                'attention_mask'][:, 0].unsqueeze(1)

            keep_decoding = [1] * item['output']['output_ids_x'].size(0)
            decoded_words = [
                [] for x in range(item['output']['output_ids_x'].size(0))
            ]
            for j in range(max_len):
                output = model_with_loss(item, mode='decode', memory=memory)
                max_ids = output[:, -1].max(-1)[1]
                item['output']['output_ids_x'] = torch.cat(
                    [item['output']['output_ids_x'],
                     max_ids.unsqueeze(1)],
                    dim=1)
                item['output']['attention_mask'] = torch.cat([
                    item['output']['attention_mask'],
                    torch.ones_like(max_ids).unsqueeze(1)
                ],
                                                             dim=1)

                for k in range(len(keep_decoding)):
                    word_id = max_ids[k].item()
                    if keep_decoding[k]:
                        cur_word = vocab.itos[word_id]
                        if cur_word != '<eos>':
                            decoded_words[k].append(cur_word)
                        else:
                            keep_decoding[k] = 0
                if max(keep_decoding) == 0:
                    break
            for k in range(len(decoded_words)):
                results.append(decoded_words[k])

        print("Decoded samples: ")
        for idx in sample_results_id:
            sample = results[idx]
            word = data_iter.dataset.raw_data[idx][0]
            sentence = ''
            for s in sample:
                if s not in vocab.stoi:
                    sentence += "[{}] ".format(s)
                else:
                    sentence += "{} ".format(s)
            print("{}\t{}".format(word, sentence))
    return results


def bleu(hyp, data_iter, bleu_path='./sentence-bleu', nltk='cpp'):
    assert nltk in ['cpp', 'corpus', 'sentence'], \
        "nltk param should be cpp/corpus/sentence"
    assert len(hyp) == len(data_iter.dataset.raw_data), \
        "sentence num in hyp not equal to dataset"
    tmp_dir = "/tmp"
    suffix = ''.join(random.sample(string.ascii_letters + string.digits, 8))
    hyp_path = os.path.join(tmp_dir, 'hyp-' + suffix)
    base_ref_path = os.path.join(tmp_dir, 'ref-' + suffix)
    to_be_deleted = set()
    to_be_deleted.add(hyp_path)

    ref_dict = defaultdict(list)
    for word, exp, sense in data_iter.dataset.raw_data:
        ref_dict[word].append(sense)

    score = 0
    num_hyp = 0
    if nltk == 'corpus':
        refs = []
    with open(os.devnull, 'w') as devnull:
        for idx, desc in enumerate(hyp):
            word = data_iter.dataset.raw_data[idx][0]
            if nltk == 'sentence':
                if len(desc) == 0:
                    auto_reweigh = False
                else:
                    auto_reweigh = True
                bleu = bleu_score.sentence_bleu(
                    [r.split(' ') for r in ref_dict[word]],
                    desc,
                    smoothing_function=bleu_score.SmoothingFunction().method2,
                    auto_reweigh=auto_reweigh)
                score += bleu
                num_hyp += 1

            elif nltk == 'corpus':
                refs.append([r.split(' ') for r in ref_dict[word]])

            elif nltk == 'cpp':
                ref_paths = []
                for i, ref in enumerate(ref_dict[word][:30]):
                    ref_path = base_ref_path + str(i)
                    with open(ref_path, 'w') as f:
                        f.write(ref + '\n')
                        ref_paths.append(ref_path)
                        to_be_deleted.add(ref_path)

                with open(hyp_path, 'w') as f:
                    f.write(' '.join(desc) + '\n')

                rp = Popen(['cat', hyp_path], stdout=PIPE)
                bp = Popen([bleu_path] + ref_paths,
                           stdin=rp.stdout,
                           stdout=PIPE,
                           stderr=devnull)
                out, err = bp.communicate()
                bleu = float(out.strip())
                score += bleu
                num_hyp += 1

            else:
                raise ValueError("nltk must be sentence/corpus/cpp")
    if nltk == 'cpp':
        for f in to_be_deleted:
            if os.path.exists(f):
                os.remove(f)
    if nltk == 'corpus':
        bleu = bleu_score.corpus_bleu(
            refs, [h for h in hyp],
            smoothing_function=bleu_score.SmoothingFunction().method2)
        ret_bleu = bleu
    else:
        ret_bleu = score / num_hyp

    return ret_bleu

def bleu_for_every_sent(hyp, data_iter, bleu_path='./sentence-bleu', nltk='cpp'):
    assert nltk in ['cpp', 'corpus', 'sentence'], \
        "nltk param should be cpp/corpus/sentence"
    assert len(hyp) == len(data_iter.dataset.raw_data), \
        "sentence num in hyp not equal to dataset"
    tmp_dir = "/tmp"
    suffix = ''.join(random.sample(string.ascii_letters + string.digits, 8))
    hyp_path = os.path.join(tmp_dir, 'hyp-' + suffix)
    base_ref_path = os.path.join(tmp_dir, 'ref-' + suffix)
    to_be_deleted = set()
    to_be_deleted.add(hyp_path)

    ref_dict = defaultdict(list)
    for word, exp, sense in data_iter.dataset.raw_data:
        ref_dict[word].append(sense)

    score = 0
    num_hyp = 0
    bleu_list = []
    with open(os.devnull, 'w') as devnull:
        for idx, desc in enumerate(hyp):
            word = data_iter.dataset.raw_data[idx][0]
            ref_paths = []
            for i, ref in enumerate(ref_dict[word][:30]):
                ref_path = base_ref_path + str(i)
                with open(ref_path, 'w') as f:
                    f.write(ref + '\n')
                    ref_paths.append(ref_path)
                    to_be_deleted.add(ref_path)

            with open(hyp_path, 'w') as f:
                f.write(' '.join(desc) + '\n')

            rp = Popen(['cat', hyp_path], stdout=PIPE)
            bp = Popen([bleu_path] + ref_paths,
                        stdin=rp.stdout,
                        stdout=PIPE,
                        stderr=devnull)
            out, err = bp.communicate()
            bleu = float(out.strip())
            score += bleu
            bleu_list.append(bleu)
            num_hyp += 1

    for f in to_be_deleted:
        if os.path.exists(f):
            os.remove(f)
    
    ret_bleu = score / num_hyp
    return ret_bleu,bleu_list