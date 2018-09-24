import pytest


import os
import json
import logging
from numpy.testing import assert_array_almost_equal, assert_almost_equal
import numpy as np
from typing import Iterator, List, Dict, Tuple

import const
import data_prepare

import torch.nn as nn
import torch
from torch.autograd import Variable
from nets import Encoder, Decoder, Seq2seq


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


# encoder = Encoder(config, load_weights=True, update_embedding=True)
# decoder = Decoder(config)

nets = Seq2seq(config)


class TestClass:

    def test1(self):
        assert 1==1

    def test2(self):

        assert len(lengths) == 100
        assert torch.Size([100, 80]) == sentence.size()

    def test3(self):
        predict = nets(sentence, lengths, triplets)
        # assert torch.Size([100, 1, 2000]) == o.size()
        # assert torch.Size([1, 100, 2000]) == h.size()
        assert predict.size() == torch.Size([100, nets.decoder.relation_number+1])


    def test4(self):
        assert nets.decoder.word_embedding(torch.zeros(100, dtype=torch.int64)).size() == torch.Size([100, 100])
    def test5(self):
        assert nets.decoder.relation_eos == 247

    def test6(self):
        assert nets.decoder.relation_embedding(torch.LongTensor([nets.decoder.relation_eos])) is not None

if __name__ == "__main__":
    print()
    # print(sentence.size())
    # print(batch.input_sentence_append_eos[2])
    # print(type(nn.Embedding(1000,50)))
    print(triplets[1])
    # print(torch.zeros(100, dtype=torch.int64))
    pytest.main('-q test_hello.py')
