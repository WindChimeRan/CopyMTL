import os
import argparse
from typing import Tuple

import torch
import torch.nn as nn

import const
import data_prepare
import evaluation

from model import Seq2seq

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', type=str, default='0', help='gpu id')
parser.add_argument('--mode', '-m', type=str, default='train', help='train/valid/test')
parser.add_argument('--cell', '-c', type=str, default='lstm', help='gru/lstm')
parser.add_argument('--decoder_type', '-d', type=str, default='multi', help='one/multi')

args = parser.parse_args()
mode = args.mode
cell_name = args.cell
decoder_type = args.decoder_type

torch.manual_seed(77) # cpu
torch.cuda.manual_seed(77) #gpu


class Evaluator(object):
    def __init__(self, config: const.Config, mode: str, device: torch.device) -> None:

        self.config = config

        self.device = device

        self.seq2seq = Seq2seq(config, device=device)

        data = prepare.load_data(mode)
        data = prepare.process(data)
        self.data = data_prepare.Data(data, config.batch_size, config)

    def load_model(self) -> None:

        # model_path = os.path.join(self.config.runner_path, 'model.pkl')
        model_path = os.path.join('saved_model', self.config.dataset_name + '_' + self.config.cell_name + '.pkl')
        self.seq2seq.load_state_dict(torch.load(model_path))

    def test_step(self, batch: data_prepare.InputData) -> Tuple[torch.Tensor, torch.Tensor]:

        sentence = batch.sentence_fw
        sentence_eos = batch.input_sentence_append_eos

        sentence = torch.from_numpy(sentence).to(self.device)
        sentence_eos = torch.from_numpy(sentence_eos).to(self.device)

        lengths = torch.Tensor(batch.input_sentence_length).int().tolist()

        pred_action_list, pred_logits_list = self.seq2seq(sentence, sentence_eos, lengths)
        pred_action_list = torch.cat(list(map(lambda x: x.unsqueeze(1), pred_action_list)), dim=1)

        return pred_action_list, pred_logits_list

    def test(self) -> Tuple[float, float, float]:

        predicts = []
        gold = []
        for batch_i in range(self.data.batch_number):
            batch_data = self.data.next_batch(is_random=False)
            pred_action_list, pred_logits_list = self.test_step(batch_data)
            pred_action_list = pred_action_list.cpu().numpy()

            predicts.extend(pred_action_list)
            gold.extend(batch_data.all_triples)

        f1, precision, recall = evaluation.compare(predicts, gold, self.config, show_rate=None, simple=True)
        return f1, precision, recall


class SupervisedTrainer(object):
    def __init__(self, config: const.Config, device: torch.device) -> None:

        self.config = config

        self.device = device

        self.seq2seq = Seq2seq(config, device=device, load_emb=True)
        self.loss = nn.NLLLoss()
        self.optimizer = torch.optim.Adam(self.seq2seq.parameters())

        data = prepare.load_data('train')
        data = prepare.process(data)
        self.data = data_prepare.Data(data, config.batch_size, config)

        self.epoch_number = config.epoch_number + 1

    def train_step(self, batch: data_prepare.InputData) -> torch.Tensor:

        self.optimizer.zero_grad()

        sentence = batch.sentence_fw
        sentence_eos = batch.input_sentence_append_eos

        triplets = batch.standard_outputs

        triplets = torch.from_numpy(triplets).to(self.device)
        sentence = torch.from_numpy(sentence).to(self.device)
        sentence_eos = torch.from_numpy(sentence_eos).to(self.device)

        lengths = torch.Tensor(batch.input_sentence_length).int().tolist()

        pred_action_list, pred_logits_list = self.seq2seq(sentence, sentence_eos, lengths)
        loss = 0
        for t in range(self.seq2seq.decoder.decodelen):
            loss = loss + self.loss(pred_logits_list[t], triplets[:, t])

        loss.backward()
        self.optimizer.step()
        return loss

    def train(self, evaluator: Evaluator=None) -> None:

        for epoch in range(1, self.epoch_number + 1):

            for step in range(self.data.batch_number):
                batch = self.data.next_batch(is_random=True)
                loss = self.train_step(batch)

            model_path = os.path.join('saved_model', self.config.dataset_name + '_' + self.config.cell_name + '.pkl')
            torch.save(self.seq2seq.state_dict(), model_path)

            if evaluator:
                evaluator.data.reset()
                evaluator.load_model()
                f1, precision, recall = evaluator.test()
                print("epoch %d \t loss: %f \t F1: %f \t P: %f \t R: %f" % (epoch, loss.item(), f1, precision, recall))


if __name__ == '__main__':

    config_filename = './config.json'
    config = const.Config(config_filename=config_filename, cell_name=cell_name, decoder_type=decoder_type)

    assert cell_name in ['lstm', 'gru']
    assert decoder_type in ['one', 'multi']

    if config.dataset_name == const.DataSet.NYT:
        prepare = data_prepare.NYTPrepare(config)
    elif config.dataset_name == const.DataSet.WEBNLG:
        prepare = data_prepare.WebNLGPrepare(config)
    else:
        print('illegal dataset name: %s' % config.dataset_name)
        exit()

    device = torch.device('cuda:' + args.gpu)

    train = True if mode == 'train' else False

    if train:
        trainer = SupervisedTrainer(config, device)
        evaluator = Evaluator(config, 'test', device)
        trainer.train(evaluator)
    else:
        tester = Evaluator(config, mode, device)
        tester.load_model()

        f1, precision, recall = tester.test()

        print(f1, precision, recall)

