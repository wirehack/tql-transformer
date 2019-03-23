import unittest
import torch.nn as nn
import torch.nn.functional as F
import torch
from src import modules


class PositionwiseFeedForwardTrue(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForwardTrue, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class TestPFF(unittest.TestCase):
    def test_equals(self):
        torch.manual_seed(0)
        myPFF = modules.PositionwiseFeedForward(512, 10, 0)
        torch.manual_seed(0)
        truePFF = PositionwiseFeedForwardTrue(512, 10, 0)
        rand_tensor = torch.rand(5, 512)
        my_output = myPFF(rand_tensor)
        true_output = truePFF(rand_tensor)
        self.assertTrue(torch.allclose(my_output, true_output))


if __name__ == '__main__':
    unittest.main()
