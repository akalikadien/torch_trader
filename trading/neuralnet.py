# -*- coding: utf-8 -*- 
# __author__: Adarsh Kalikadien #

import torch
import torch.nn as nn
import torch.nn.functional as F


class QvalueNN(nn.Module):
    def __init__(self, state_size, action_size, units):
        super(QvalueNN, self).__init__()
        self._state_size = state_size
        self._action_size = action_size
        self._units = units
        self.fc1 = nn.Linear(self._state_size, self._units)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self._units, self._action_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


if __name__ == '__main__':
    test = QvalueNN(9, 3, 9)
    print(test)
