# -*- coding: utf-8 -*-
import torch
import math
import torch.nn as nn
import numpy as np
from transformers import AutoModel
from src import utils


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe

    def forward(self, x):
        pe = torch.FloatTensor(self.pe[:, :x.size(1)]).to(x.device)
        pe.requires_grad = False
        x = x + pe
        return x


class MyTransformerDecoder(nn.Module):
    def __init__(self,
                 d_memory,
                 len_vocab,
                 d_model,
                 nhead,
                 nlayer,
                 tie_readout=True,
                 dim_ff=2048,
                 dropout=0.1,
                 activation='relu'):
        super(MyTransformerDecoder, self).__init__()
        dec_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_ff, dropout,
                                               activation)
        dec_norm = nn.LayerNorm(d_model)
        dec_layer.apply(utils.weights_init)

        self.d_model = d_model
        self.d_memory = d_memory
        self.tgt_emb = nn.Embedding(len_vocab, d_model)
        self.tgt_pos = PositionalEncoding(d_model)
        self.tgt_emb_layernorm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.memory_proj = nn.Linear(d_memory, d_model)
        self.memory_proj_layernorm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(dec_layer, nlayer, dec_norm)
        self.out_proj = nn.Linear(d_model, len_vocab, bias=False)

        utils.weights_init(self.memory_proj)
        if tie_readout:
            self.out_proj.weight = self.tgt_emb.weight
        else:
            utils.weights_init(self.out_proj)

    def init_embedding(self, emb_matrix, fix_embedding=False):
        self.tgt_emb.weight.data.copy_(torch.from_numpy(emb_matrix))
        if fix_embedding:
            self.tgt_emb.weight.requires_grad = False

    def forward(self,
                tgt,
                tgt_mask,
                tgt_key_padding_mask,
                memory,
                memory_key_padding_mask):
        mapped_memory = self.drop(
            self.memory_proj_layernorm(self.memory_proj(memory)))

        tgt_vec = self.drop(
            self.tgt_emb_layernorm(
                self.tgt_pos(self.tgt_emb(tgt) * (self.d_memory**0.5))))
        
        output_sense = self.decoder(
            tgt_vec.permute(1, 0, 2),
            mapped_memory.permute(1, 0, 2),
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask).permute(1, 0, 2)
        output = torch.log_softmax(self.out_proj(output_sense), dim=-1)
        return output


class MyModel(nn.Module):
    def __init__(self,
                 enc_arch,
                 enc_d_model,
                 len_vocab,
                 dec_d_model,
                 dec_nhead,
                 dec_nlayer,
                 tie_readout,
                 dec_d_ff,
                 dec_dropout,
                 dec_activation,
                 vector_path=None,
                 fix_embedding=False,
                 use_pretrained_encoder=False,
                 fix_encoder=False,
                 exp_mask=False):
        super(MyModel, self).__init__()
        config = utils.obtain_config(enc_arch)
        self.enc_model = AutoModel.from_config(config)
        self.dec_model = MyTransformerDecoder(enc_d_model,
                                              len_vocab,
                                              dec_d_model,
                                              dec_nhead,
                                              dec_nlayer,
                                              tie_readout=tie_readout,
                                              dim_ff=dec_d_ff,
                                              dropout=dec_dropout,
                                              activation=dec_activation)
        if vector_path:
            emb_matrix = np.load(vector_path)
            self.dec_model.init_embedding(emb_matrix, fix_embedding)
        self.enc_arch = enc_arch
        self.exp_mask = exp_mask
        if use_pretrained_encoder:
            self.init_encoder(fix_encoder)

    def init_encoder(self, fix_encoder=False):
        arch_dict = {
            "enbert": "bert-base-uncased",
            "cbert": "bert-base-chinese"
        }
        self.enc_model = self.enc_model.from_pretrained(
            arch_dict[self.enc_arch])
        if fix_encoder:
            for p in self.enc_model.parameters():
                p.requires_grad = False

    def forward(self, item, mode=None, memory=None):
        assert mode in [None, 'encode', 'decode'], \
            "mode must be None/encode/decode"
        if mode == 'encode':
            return self.encode(item)
        elif mode == "decode":
            assert memory is not None, \
                "Memory is not None when decoding"
            return self.decode(item, memory)
        else:
            return self.decode(item, self.encode(item))

    def encode(self, item):
        # import pdb;pdb.set_trace()
        memory = self.enc_model(
                input_ids=item['input']['input_ids'],
                attention_mask=item['input']['attention_mask'],
                token_type_ids=item['input']['token_type_ids'],
                position_ids=item['input']['position_ids'])[0]
        return memory

    def decode(self, item, memory):
        subsequent_mask = utils.generate_square_subsequent_mask(
            item['output']['output_ids_x'].size(1)).to(
                item['output']['output_ids_x'].device)
        tgt_key_padding_mask = (item['output']['attention_mask'] == 0)
        if self.exp_mask:
            memory_key_padding_mask = (item['input']['exp_mask'] == 0)
        else:
            memory_key_padding_mask = (item['input']['attention_mask'] == 0)

        output = self.dec_model(item['output']['output_ids_x'],
                                subsequent_mask,
                                tgt_key_padding_mask,
                                memory,
                                memory_key_padding_mask)
        return output


class ModelWithLoss(nn.Module):
    def __init__(self,
                 model,
                 loss_ce,
                 loss):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss_ce = loss_ce
        self.loss = loss

    def forward(self, item, mode=None, memory=None):
        output = self.model(item, mode, memory)
        if not mode:
            output_sense = output
            output_sense_flat = output_sense.view(
                output_sense.size(0) * output_sense.size(1), -1)
            target = item['output']['output_ids_y']
            loss = self.loss(output_sense_flat, target.reshape(-1))
            with torch.no_grad():
                loss_ce = self.loss_ce(output_sense_flat, target.reshape(-1))
            return output, loss_ce.unsqueeze(0), loss.unsqueeze(0)
        else:
            return output
