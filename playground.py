import os
import json
import logging
import numpy as np
from typing import Iterator, List, Dict, Tuple

import const
import data_prepare

import torch.nn as nn
import torch
from torch.autograd import Variable

logger = logging.getLogger('mylogger')


class Encoder(nn.Module):
    def __init__(self, config: const.Config, load_weights=True, update_embedding=True) -> None:
        super(Encoder, self).__init__()
        self.config = config

        self.hidden_size = config.encoder_num_units
        self.emb_size = config.embedding_dim
        self.words_number = config.words_number

        self.embedding = nn.Embedding(self.words_number, self.emb_size)
        if load_weights:
            self.load_pretrain_emb(config)
        self.embedding.weight.requires_grad = update_embedding
        self.rnn = nn.GRU(self.emb_size, self.hidden_size, bidirectional=True, batch_first=True)

    def forward(self, sentence: torch.Tensor, lengths: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:

        embedded = self.embedding(sentence)
        print(embedded.size())
        if lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths=lengths, batch_first=True)

        output, hidden = self.rnn(embedded)

        if lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        hidden = hidden.view(-1, self.hidden_size * 2)

        return output, hidden

    def load_pretrain_emb(self, config: const.Config) -> None:
        if os.path.isfile(config.words_id2vector_filename):
            logger.info('Word Embedding init from %s' % config.words_id2vector_filename)
            words_id2vec = json.load(open(config.words_id2vector_filename, 'r'))
            words_vectors = [0] * len(words_id2vec)

            for i, key in enumerate(words_id2vec):
                words_vectors[int(key)] = words_id2vec[key]

            self.embedding.weight.data.copy_(torch.from_numpy(np.array(words_vectors)))


class Decoder(nn.Module):
    def __init__(self):
        pass

    def forward(self, *input):
        pass


config_filename = './config.json'
cell_name = 'gru'
config = const.Config(config_filename=config_filename, cell_name=cell_name)
if config.dataset_name == const.DataSet.NYT:
    prepare = data_prepare.NYTPrepare(config)
elif config.dataset_name == const.DataSet.WEBNLG:
    prepare = data_prepare.WebNLGPrepare(config)
else:
    print('illegal dataset name: %s' % config.dataset_name)
    exit()

data = prepare.load_data('train')
# sentence, triplets = data
data = prepare.process(data)
data = data_prepare.Data(data, config.batch_size, config)
# data.all_sentence = sentence
batch = data.next_batch(is_random=True)

sentence = batch.sentence_fw
triplets = batch.standard_outputs

sentence = torch.from_numpy(sentence)

lengths = torch.Tensor(batch.input_sentence_length).int().tolist()


encoder = Encoder(config, load_weights=True)
o, h = encoder(sentence, lengths)
print(o.size())
print(h.size())
# print(encoder.embedding.weight.data[1])
# print(sentence.shape)
# print(lengths)


# print(triplets[10])
# print((triplets.shape))
# print((sentence.shape))
# sentence_fw = batch.sentence_fw
# sentence_bw = batch.sentence_bw
# print(batch.all_triples.shape)
# print(batch.all_triples[1])
# print(batch.sentence_pos_fw.shape)

