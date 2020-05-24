'''
net test 是用于将预训练好的单车和双车model加载进来，输出动作和Q值，不用再训练了
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.95                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000


class Net_10dim(nn.Module):
    def __init__(self):
        super(Net_10dim, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(50, 4)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = F.relu(self.fc1(x))
        actions_value = self.out(x)
        return actions_value

class Net_5dim(nn.Module):
    def __init__(self):
        super(Net_5dim, self).__init__()
        self.fc1 = nn.Linear(5, 50)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(50, 2)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = F.relu(self.fc1(x))
        actions_value = self.out(x)
        return actions_value

class DQN_10dim_test(object):
    def __init__(self):
        self.eval_net, self.target_net = Net_10dim(), Net_10dim()
        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        # s, [a, r], s_
        self.memory = np.zeros((MEMORY_CAPACITY, 10 * 2 + 2))  # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_actions(self, x, epsilon):
        # input only one sample
        # Note: Action range: 0 1 2 3
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() <= epsilon:  # greedy
            actions_value = self.eval_net.forward(x)
            # print('actions_value',actions_value)
            maxQ = str(torch.max(actions_value, 1)[0].data.numpy()[0])
            #with open('../data/Q10dim.txt', 'a') as file_handle:  # .txt可以不自己新建,代码会自动新建
            #    file_handle.write(datastr)  # 写入
            #    file_handle.write('\n')

            # choose action
            action = torch.max(actions_value, 1)[1].data.numpy()


        # 返回的是N_action的list
        return action, maxQ

class DQN_5dim_test(object):
    def __init__(self):
        self.eval_net, self.target_net = Net_5dim(), Net_5dim()
        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        # s, [a, r], s_
        self.memory = np.zeros((MEMORY_CAPACITY, 5 * 2 + 2))  # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_actions(self, x, epsilon):
        # input only one sample
        # Note: Action range: 0 1

        # open txt

        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() <= epsilon:  # greedy
            actions_value = self.eval_net.forward(x)
            # print('actions_value',torch.max(actions_value, 1)[0].data.numpy()[0])
            maxQ = str(torch.max(actions_value, 1)[0].data.numpy()[0])
            #with open('../data/Q5dim.txt', 'a') as file_handle:  # .txt可以不自己新建,代码会自动新建
            #    file_handle.write(datastr)  # 写入
            #    file_handle.write('\n')
            action = torch.max(actions_value, 1)[1].data.numpy()


        # 返回的是N_action的list
        return action, maxQ



