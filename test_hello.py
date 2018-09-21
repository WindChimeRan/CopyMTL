import pytest
import torch


class Testcase001:
    def test_001(self):
        x = "hello"
        assert "h" in x

    def test_002(self):
        x = "this"
        assert 't' in x

    def test_003(self):
        a = 10
        assert a < 20

    def test_004(self):
        assert torch.cuda.is_available() == True

    def test_005(self):
        import torch
        import numpy as np

        input = torch.from_numpy(np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 0]]))
        length = [4,4,4]  # lengths array has to be sorted in decreasing order
        result = torch.nn.utils.rnn.pack_padded_sequence(input, lengths=length, batch_first=True)
        print(result)

        input = torch.randn(8, 10, 300)
        length = [10, 10, 10, 10, 10, 10, 10, 10]
        perm = torch.LongTensor(range(8))
        result = torch.nn.utils.rnn.pack_padded_sequence(input[perm], lengths=length, batch_first=True)
        print(result)