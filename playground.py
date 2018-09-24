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
from nets import Encoder, Decoder


if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device

logger = logging.getLogger('mylogger')


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
data = prepare.process(data)
data = data_prepare.Data(data, config.batch_size, config)
batch = data.next_batch(is_random=True)

sentence = batch.sentence_fw
triplets = batch.standard_outputs

sentence = torch.from_numpy(sentence)
lengths = torch.Tensor(batch.input_sentence_length).int().tolist()


encoder = Encoder(config)
decoder = Decoder(config)
o, h = encoder(sentence, lengths)
decoder(token=None, decoder_state=h, encoder_outputs=o)
# print(o.size())
# print(h.size())
# print(encoder.word_embedding.weight.data[1])
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

