import numpy as np
from typing import Iterator, List, Dict, Tuple
import os
import const
import json

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device


class Encoder(nn.Module):
    def __init__(self, config: const.Config, embedding) -> None:
        super(Encoder, self).__init__()
        self.config = config

        self.hidden_size = config.encoder_num_units
        self.emb_size = config.embedding_dim
        self.words_number = config.words_number
        self.maxlen = config.max_sentence_length

        self.embedding = embedding
        self.rnn = nn.GRU(self.emb_size, self.hidden_size, bidirectional=True, batch_first=True)

    def forward(self, sentence: torch.Tensor, lengths: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:

        embedded = self.embedding(sentence)
        # print(embedded.size())
        if lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths=lengths, batch_first=True)

        output, hidden = self.rnn(embedded)

        if lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, total_length=self.maxlen, batch_first=True)

        hidden = hidden.view(-1, self.hidden_size * 2)

        return output, hidden



class Decoder(nn.Module):
    def __init__(self, config: const.Config, embedding):
        super(Decoder, self).__init__()

        self.hidden_size = config.decoder_num_units
        self.emb_size = config.embedding_dim
        self.words_number = config.words_number
        self.maxlen = config.max_sentence_length

        self.relation_eos = config.relation_number
        self.relation_number = config.relation_number
        self.word_embedding = embedding
        self.relation_embedding = nn.Embedding(config.relation_number + 1, config.embedding_dim)


        self.combine_inputs = nn.Linear(2 * self.hidden_size + self.emb_size, self.emb_size)
        self.attn = nn.Linear(self.hidden_size * 4, 1)
        self.rnn = nn.GRU(self.emb_size, 2 * self.hidden_size, batch_first=True)
        self.do_eos = nn.Linear(2 * self.hidden_size, 1)
        self.do_predict = nn.Linear(2 * self.hidden_size, self.relation_number)

    def forward_step(self):
        pass


    def calc_context(self, decoder_state, encoder_outputs):

        attn_weight = torch.cat((decoder_state.unsqueeze(1) + torch.zeros_like(encoder_outputs), encoder_outputs), dim=2)
        attn_weight = attn_weight.view(-1, 4 * self.hidden_size)
        attn_weight = F.softmax(F.selu(self.attn(attn_weight).view(-1, self.maxlen)), dim=1)

        attn_applied = torch.bmm(attn_weight.unsqueeze(1), encoder_outputs).squeeze(1)

        # I wanna add some selu here..., however, ahhh!! just exactly match the paper!
        return attn_applied

    def _decode_step(self, token, decoder_state, encoder_outputs):

        context = self.calc_context(decoder_state, encoder_outputs)
        output =  self.combine_inputs(torch.cat((self.word_embedding(token), context), dim=1))
        output, decoder_state = self.rnn(output.unsqueeze(1), decoder_state.unsqueeze(0))

        output = output.squeeze()

        eos_logits = F.selu(self.do_eos(output))
        predict_logits = F.selu(self.do_predict(output))

        predict = F.softmax(torch.cat((predict_logits, eos_logits), dim=1), dim=1)

        # TODO copy!
        return predict

    def forward(self, token, decoder_state, encoder_outputs):
        # sos = go = 0

        go = torch.zeros(decoder_state.size()[0], dtype=torch.int64)

        out = self._decode_step(go, decoder_state, encoder_outputs)
        # context = self.calc_context(decoder_state, encoder_outputs)
        # go = 0
        return out

class Seq2seq(nn.Module):
    def __init__(self, config:const.Config, load_emb=True, update_emb=True):
        super(Seq2seq, self).__init__()

        self.emb_size = config.embedding_dim
        self.words_number = config.words_number
        self.maxlen = config.max_sentence_length

        self.word_embedding = nn.Embedding(self.words_number, self.emb_size)
        if load_emb:
            self.load_pretrain_emb(config)
        self.word_embedding.weight.requires_grad = update_emb

        self.encoder = Encoder(config, embedding=self.word_embedding)
        self.decoder = Decoder(config, embedding=self.word_embedding)

    def load_pretrain_emb(self, config: const.Config) -> None:
        if os.path.isfile(config.words_id2vector_filename):
            # logger.info('Word Embedding init from %s' % config.words_id2vector_filename)
            words_id2vec = json.load(open(config.words_id2vector_filename, 'r'))
            words_vectors = [0] * len(words_id2vec)

            for i, key in enumerate(words_id2vec):
                words_vectors[int(key)] = words_id2vec[key]

            self.word_embedding.weight.data.copy_(torch.from_numpy(np.array(words_vectors)))

    def forward(self, sentence, lengths, triplets):
        o, h = self.encoder(sentence, lengths)
        context = self.decoder(token=None, decoder_state=h, encoder_outputs=o)
        return context
