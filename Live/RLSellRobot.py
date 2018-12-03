#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 11:43:19 2018

@author: lipchiz
"""

import os
import sys
# os.chdir(r'/home/lipchiz/Documents/pythonscripts/quant/DQN/test')
import numpy as np
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 1)
        self.bin1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 1)
        self.bin2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, 1)
        self.bin3 = nn.BatchNorm2d(16)
        
        self.line1 = nn.Linear(640, 320)
        self.line2 = nn.Linear(320, 2)
        
    def forward(self, x):
        x = F.relu(self.bin1(self.conv1(x)))
        x = F.relu(self.bin2(self.conv2(x)))
        x = F.relu(self.bin3(self.conv3(x)))
        x = x.view(-1, 16*10*4)  # 展平多8维的卷积图成 (batch_size, 16 * 10 * 4)
        x = self.line1(x)
        x = self.line2(x)
        return x

def choose_action(x):
    """
    input:
    x: 10*4 proba matrix
    
    output:
    action: sell the stocks 1
    """
    # model_file_path = '/data2/jianghan/FeatureAlgorithm/Live/target_net_param_priority_2014_cpu.json'
    model_file_path = 'target_net_param_priority_2014.json'

    with open(model_file_path, 'rb+') as f:
        target_net_param = pickle.load(f)

#构造神经网络

    target_net = Net()
    target_net.load_state_dict(target_net_param)
    x = x[np.newaxis, np.newaxis, :, :]
    x = torch.Tensor(x)
    action = target_net.forward(x)
    action = int(action.argmax().numpy())
    return action