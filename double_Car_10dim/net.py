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


class DQN_10dim(object):
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
        if np.random.uniform() < epsilon:  # greedy
            actions_value = self.eval_net.forward(x)
            # print('actions_value',actions_value)
            datastr = str(torch.max(actions_value, 1)[0].data.numpy()[0])
            with open('../data/Q10dim.txt', 'a') as file_handle:  # .txt可以不自己新建,代码会自动新建
                file_handle.write(datastr)  # 写入
                file_handle.write('\n')

            # choose action
            action = torch.max(actions_value, 1)[1].data.numpy()

        else:  # random
            # action = [np.random.randint(0, N_ACTIONS) for _ in range(N_AGENTS)]
            action = np.random.randint(0, 4)
            action = [action]

            # 为做存储用，特意过一遍神经网络，存储actions_value,实际使用的动作还是随机的
            actions_value = self.eval_net.forward(x)
            datastr = str(torch.max(actions_value, 1)[0].data.numpy()[0])
            with open('../data/Q10dim.txt', 'a') as file_handle:  # .txt可以不自己新建,代码会自动新建
                file_handle.write(datastr)  # 写入
                file_handle.write('\n')
        # 返回的是N_action的list
        return action

    def store_transition(self, s, a, r, s_):
        # transform
        s = np.array(s)
        s_ = np.array(s_)
        a = np.array(a)
        r = np.array(r)
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE, replace=False)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :10])
        b_a = torch.LongTensor(b_memory[:, 10:10 + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, 10 + 1:10 + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -10:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class DQN_5dim(object):
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
        if np.random.uniform() < epsilon:  # greedy
            actions_value = self.eval_net.forward(x)
            # print('actions_value',torch.max(actions_value, 1)[0].data.numpy()[0])
            datastr = str(torch.max(actions_value, 1)[0].data.numpy()[0])
            with open('../data/Q5dim.txt', 'a') as file_handle:  # .txt可以不自己新建,代码会自动新建
                file_handle.write(datastr)  # 写入
                file_handle.write('\n')
            action = torch.max(actions_value, 1)[1].data.numpy()
        else:  # random
            # action = [np.random.randint(0, N_ACTIONS) for _ in range(N_AGENTS)]
            action = np.random.randint(0, 2)

            # 为做存储用，特意过一遍神经网络，存储actions_value,实际使用的动作还是随机的
            actions_value = self.eval_net.forward(x)
            datastr = str(torch.max(actions_value, 1)[0].data.numpy()[0])
            with open('../data/Q5dim.txt', 'a') as file_handle:  # .txt可以不自己新建,代码会自动新建
                file_handle.write(datastr)  # 写入
                file_handle.write('\n')

        # 返回的是N_action的list
        return action

    def store_transition(self, s, a, r, s_):
        # transform
        s = np.array(s)
        s_ = np.array(s_)
        a = np.array(a)
        r = np.array(r)
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE, replace=False)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :5])
        b_a = torch.LongTensor(b_memory[:, 5:5 + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, 5 + 1:5 + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -5:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


