from car import Car
from state import State
import random
from net import DQN_5dim
import torch
from GUI import GUI
import pygame
import time
import csv


class Environment(object):
    # 一些参数

    # 前向感知距离
    fv = 160
    # 后向感知距离
    bv = 160.0
    # 当前动作全局奖励值
    r = 0.0
    # episode总奖励值
    reward = 0

    def __init__(self):
        # some attribute
        # 该步时全局奖励值
        self.r = 0.0
        # 一个episode总奖励值
        self.episode_reward = 0.0
        # 总episode
        self.episode = 0

        # about init car
        # Car(car id, no, lane, velocity, position)
        self.car_list = []
        auto_car = Car(0, 0, 1, random.randint(18, 27), 200)
        self.car_list.append(auto_car)
        en_car1 = Car(1, 1, 1, random.randint(18, 27), 200 + random.randint(10, 25) * 10)
        self.car_list.append(en_car1)
        en_car2 = Car(2, 2, 1, random.randint(18, 27), 200 - random.randint(10, 25) * 10)
        self.car_list.append(en_car2)
        en_car3 = Car(3, 3, 2, random.randint(18, 27), 200 + random.randint(10, 25) * 10)
        self.car_list.append(en_car3)
        en_car4 = Car(4, 4, 2, random.randint(18, 27), 200 - random.randint(10, 25) * 10)
        self.car_list.append(en_car4)
        en_car5 = Car(5, 5, 2, random.randint(18, 27), 200 + random.randint(10, 25))
        self.car_list.append(en_car5)

    def add_car(self):
        # 初始化训练
        self.car_list[0].car_id = 0
        self.car_list[0].no = 0
        self.car_list[0].lane = 1
        self.car_list[0].a = 1
        self.car_list[0].v0 = random.randint(18, 27)
        self.car_list[0].position = 200
        self.car_list[0].auto_a = 0
        self.car_list[1].car_id = 1
        self.car_list[1].no = 1
        self.car_list[1].lane = 1
        self.car_list[1].a = 1
        self.car_list[1].v0 = random.randint(18, 27)
        self.car_list[1].position = 200 + random.randint(10, 25) * 10
        self.car_list[1].auto_a = 0
        self.car_list[2].car_id = 2
        self.car_list[2].no = 2
        self.car_list[2].lane = 1
        self.car_list[2].a = 1
        self.car_list[2].v0 = random.randint(18, 27)
        self.car_list[2].position = 200 - random.randint(10, 25) * 10
        self.car_list[2].auto_a = 0
        self.car_list[3].car_id = 3
        self.car_list[3].no = 3
        self.car_list[3].lane = 2
        self.car_list[3].a = 2
        self.car_list[3].v0 = random.randint(18, 27)
        self.car_list[3].position = 200 + random.randint(10, 25) * 10
        self.car_list[3].auto_a = 0
        self.car_list[4].car_id = 4
        self.car_list[4].no = 4
        self.car_list[4].lane = 2
        self.car_list[4].a = 2
        self.car_list[4].v0 = random.randint(18, 27)
        self.car_list[4].position = 200 - random.randint(10, 25) * 10
        self.car_list[4].auto_a = 0
        # en_car5 = Car(5, 5, 2, random.randint(18, 27), 200+random.randint(10, 25))
        self.car_list[5].auto_a = 0
        self.car_list[5].car_id = 5
        self.car_list[5].no = 5
        self.car_list[5].lane = 2
        self.car_list[5].a = 2
        self.car_list[5].v0 = random.randint(18, 27)
        self.car_list[5].position = 200 + random.randint(10, 25)

    def get_car1(self, car_id):
        # 获取id为id的车辆行车道上前方车辆的距离d1和速度v1
        index = car_id
        d1 = self.fv
        for car in self.car_list:
            if car.car_id == car_id:
                continue
            if car.lane == 1 and car.position >= self.car_list[car_id].position:
                if car.position - self.car_list[car_id].position < d1:
                    d1 = car.position - self.car_list[car_id].position
                    index = car.car_id
        return d1, self.car_list[index].v0

    def get_car2(self, car_id):
        # 获取id为id的车辆行车道上后方车辆的距离d2和速度v2
        index = car_id
        d2 = self.bv
        for car in self.car_list:
            if car.car_id == car_id:
                continue
            if car.lane == 1 and car.position <= self.car_list[car_id].position:
                if self.car_list[car_id].position - car.position < d2:
                    d2 = self.car_list[car_id].position - car.position
                    index = car.car_id
        return d2, self.car_list[index].v0

    def get_car3(self, car_id):
        # 获取编号为no的车辆超车道上前方车辆的距离d3和速度v3
        index = car_id
        d3 = self.fv
        for car in self.car_list:
            if car.car_id == car_id:
                continue
            if car.lane == 2 and car.position >= self.car_list[car_id].position:
                if car.position - self.car_list[car_id].position < d3:
                    d3 = car.position - self.car_list[car_id].position
                    index = car.car_id
        return d3, self.car_list[index].v0

    def get_car4(self, car_id):
        # 获取编号为no的车辆超车道上后方车辆的距离d4和速度v4
        index = car_id
        d4 = self.bv
        for car in self.car_list:
            if car.car_id == car_id:
                continue
            if car.lane == 2 and car.position <= self.car_list[car_id].position:
                if self.car_list[car_id].position - car.position < d4:
                    d4 = self.car_list[car_id].position - car.position
                    index = car.car_id
        return d4, self.car_list[index].v0

    def get_dm(self, v0, v1):
        a0 = 4
        a1 = 6
        dm = v0 * v0 / (2 * a0) - v1 * v1 / (2 * a1) + 10
        return dm

    def get_state(self, car_id):
        # 或者编号为no的车辆的当前状态， 五维状态
        # 初始化状态
        s = State()

        # 取car所在车道作为状态的第一个维度
        s.l = self.car_list[car_id].lane

        # 获取行车道上前方车辆距离和速度
        d1, v1 = self.get_car1(car_id)

        # 获取行车道上后方车辆距离和速度
        d2, v2 = self.get_car2(car_id)

        # 获取超车道上前方车辆距离和速度
        d3, v3 = self.get_car3(car_id)

        # 获取超车道上后方车辆距离和速度
        d4, v4 = self.get_car4(car_id)

        dm1 = self.get_dm(self.car_list[car_id].v0, v1)
        dm2 = self.get_dm(v2, self.car_list[car_id].v0)
        dm3 = self.get_dm(self.car_list[car_id].v0, v3)
        dm4 = self.get_dm(v4, self.car_list[car_id].v0)

        s.t1 = (d1 - dm1) / self.car_list[car_id].v0 if d1 < 160 else 40
        s.t2 = (d2 - dm2) / v2 if d2 < 160 else 40
        s.t3 = (d3 - dm3) / self.car_list[car_id].v0 if d3 < 160 else 20
        s.t4 = (d4 - dm4) / v4 if d4 < 160 else 20

        return s

    def print_train_info(self):
        print("第", self.episode, "次训练:")

    def check_conflict(self):
        is_conflict = False
        for car in self.car_list:
            if car.check_conflict(self.car_list):
                is_conflict = True
                break
        return is_conflict

    def reset(self):
        # 重置步数
        self.epi_step = 0
        # 重置所有自主车的数据
        self.add_car()
        # 重置奖励
        self.episode_reward = 0
        # lane
        self.change_lane = 0
        # r0仅用于测试，没用的时候记得关
        self.r0 = 0
        # print info
        # print(self.car_list[0].print_info())
        # print(self.car_list[1].print_info())
        # print(self.car_list[2].print_info())
        # print(self.car_list[3].print_info())
        # print(self.car_list[4].print_info())

    def get_reward(self, car_id):
        for car in self.car_list:
            car.s = self.get_state(car.car_id)  # 更新每个车辆的状态信息
        s = self.car_list[car_id].s

        # reward具体需要调整
        r_safe = -5
        d1, v1 = self.get_car1(car_id)
        d2, v2 = self.get_car2(car_id)
        d3, v3 = self.get_car3(car_id)
        d4, v4 = self.get_car4(car_id)
        if s.l == 1 and d1 > 13 and d2 > 13:
            r_safe = min(s.t1, s.t2)
        elif s.l == 2 and d3 > 13 and d4 > 13:
            r_safe = min(s.t3, s.t4)
        if self.car_list[car_id].check_conflict(self.car_list):
            r_safe = -20
        return r_safe * 0.00125  # !!

    def step(self, actions):

        # 选择动作，变更车道
        # print('actions:',actions)
        for i in range(len(self.car_list)):
            self.car_list[i].a = actions[i] + 1  # action 1 2

        # 按限速跟驰策略update auto_a，如果换道则改变lane属性，如果不换道则跟随前车
        for i in range(len(self.car_list)):
            self.update_auto_a(i)
            # self.car_list[i].print_info()

        # update velocity and position
        for car in self.car_list:
            car.update_v0()
            car.update_position()

        # calculate reward
        reward = [0 for _ in range(len(self.car_list))]  # 储存当前每个车在该动作下的全局奖励
        self.r = 0  # 单步奖励和初始化
        for i in range(len(self.car_list)):
            reward[i] = self.get_reward(i)  # get reward 返回当前的该车reward值
            self.r += reward[i]  # 单步奖励和
        self.episode_reward += self.r  # 整局奖励
        self.r0 += reward[0]

        observations_ = [[] for _ in range(len(self.car_list))]
        i = 0
        for car in self.car_list:
            car.s = self.get_state(car.car_id)  # 获取每个车辆的状态信息
            observations_[i] = [car.s.l, car.s.t1, car.s.t2, car.s.t3, car.s.t4]
            i += 1

        # is done
        done = [0 for _ in range(len(self.car_list))]
        if self.check_conflict() or self.epi_step >= 400:
            done = [1 for _ in range(len(self.car_list))]

        return observations_, reward, done

    def update_auto_a(self, car_id):
        car = self.car_list[car_id]
        if car.lane == 1 and car.a == 1:  # 跟随
            v0 = car.v0
            d1, v1 = self.get_car1(car_id)
            d_follow = self.get_distance_follow(v0)
            if d1 - d_follow > 10:
                if v1 - v0 >= 1 or v1 - v0 <= -1:
                    self.car_list[car_id].v_plan = v0 + 0.25 * (d1 - d_follow) / 3.6 + 1.5 * (v1 - v0)
                else:
                    self.car_list[car_id].v_plan = v0 + 0.25 * (d1 - d_follow) / 3.6 + 1.0 * (v1 - v0)
            elif d1 - d_follow > -4:
                if v1 - v0 >= 1 or v1 - v0 <= -1:
                    self.car_list[car_id].v_plan = v0 + 0.5 * (d1 - d_follow) / 3.6 + 1.5 * (v1 - v0)
                else:
                    self.car_list[car_id].v_plan = v0 + 0.5 * (d1 - d_follow) / 3.6 + 1.0 * (v1 - v0)
            else:
                if v1 - v0 >= 1 or v1 - v0 <= -1:
                    self.car_list[car_id].v_plan = v0 + (d1 - d_follow) / 3.6 + 1.5 * (v1 - v0)
                else:
                    self.car_list[car_id].v_plan = v0 + (d1 - d_follow) / 3.6 + 1.0 * (v1 - v0)
            if self.car_list[car_id].v_plan > v1 + self.car_list[car_id].v_bound:
                self.car_list[car_id].v_plan = v1 + self.car_list[car_id].v_bound
            if self.car_list[car_id].v_plan < v1 - self.car_list[car_id].v_bound:
                self.car_list[car_id].v_plan = v1 - self.car_list[car_id].v_bound
            if self.car_list[car_id].v_plan > self.car_list[car_id].v_task:
                self.car_list[car_id].v_plan = self.car_list[car_id].v_task
        elif car.s.l == 1 and car.a == 2:  # 换道
            self.change_lane += 1
            self.car_list[car_id].lane = 2
            self.car_list[car_id].v_plan = self.car_list[car_id].v_overtake
        elif car.s.l == 2 and car.a == 1:  # 换道
            self.change_lane += 1
            self.car_list[car_id].lane = 1
            self.car_list[car_id].v_plan = self.car_list[car_id].v_task
        elif car.s.l == 2 and car.a == 2:  # 跟随
            v0 = car.v0
            d3, v3 = self.get_car3(car_id)
            d_follow = self.get_distance_follow(v0)
            if d3 - d_follow > 10:
                if v3 - v0 >= 1 or v3 - v0 <= -1:
                    self.car_list[car_id].v_plan = v0 + 0.25 * (d3 - d_follow) / 3.6 + 1.5 * (v3 - v0)
                else:
                    self.car_list[car_id].v_plan = v0 + 0.25 * (d3 - d_follow) / 3.6 + 1.0 * (v3 - v0)
            elif d3 - d_follow > -4:
                if v3 - v0 >= 1 or v3 - v0 <= -1:
                    self.car_list[car_id].v_plan = v0 + 0.5 * (d3 - d_follow) / 3.6 + 1.5 * (v3 - v0)
                else:
                    self.car_list[car_id].v_plan = v0 + 0.5 * (d3 - d_follow) / 3.6 + 1.0 * (v3 - v0)
            else:
                if v3 - v0 >= 1 or v3 - v0 <= -1:
                    self.car_list[car_id].v_plan = v0 + (d3 - d_follow) / 3.6 + 1.5 * (v3 - v0)
                else:
                    self.car_list[car_id].v_plan = v0 + (d3 - d_follow) / 3.6 + 1.0 * (v3 - v0)
            if self.car_list[car_id].v_plan > v3 + self.car_list[car_id].v_bound:
                self.car_list[car_id].v_plan = v3 + self.car_list[car_id].v_bound
            if self.car_list[car_id].v_plan < v3 - self.car_list[car_id].v_bound:
                self.car_list[car_id].v_plan = v3 - self.car_list[car_id].v_bound
            if self.car_list[car_id].v_plan > self.car_list[car_id].v_overtake:
                self.car_list[car_id].v_plan = self.car_list[car_id].v_overtake
        self.car_list[car_id].auto_a = (self.car_list[car_id].v_plan - self.car_list[car_id].v0) / 1

    def get_distance_follow(self, v0):  # 计算跟随距离
        a0 = 6  # a0为制动减速度
        d_follow = v0 * v0 / (2 * a0) + 5
        return d_follow

    def choose_expert_actions(self, car_id):
        d1, v1 = self.get_car1(car_id)
        d2, v2 = self.get_car2(car_id)
        d3, v3 = self.get_car3(car_id)
        d4, v4 = self.get_car4(car_id)

        if self.car_list[car_id].lane == 1:
            if (d1 < 50 or v1 - self.car_list[car_id].v_task < -3) \
                    and (d3 - d1 > 8 and v3 - v1 > 1) \
                    and d4 > 60:
                action = 1
            else:
                action = 0
        else:
            if (d1 > 100 or v1 > v3) and d2 > 60:
                action = 0
            else:
                action = 1

        return action

    def rl_method(self):
        # 当前回合数
        self.episode = 0

        # some parameter
        self.max_episode = 1000
        self.max_step_in_every_episode = 400
        MEMORY_SIZE = 2000

        # net
        self.net = DQN_5dim()

        # GUI
        self.GUI = GUI()

        # epsilon
        self.epsilon = 0.9

        for i_episode in range(self.max_episode):
            # 新的episode
            self.episode += 1

            # reset and print_info
            self.reset()

            self.print_train_info()

            # GUI reset
            self.GUI.reset()

            # episode running
            while True:
                # step ++
                self.epi_step += 1

                # run step
                s = [[] for _ in range(len(self.car_list))]
                i = 0
                for car in self.car_list:
                    car.s = self.get_state(car.car_id)  # 获取每个车辆的状态信息
                    s[i] = [car.s.l, car.s.t1, car.s.t2, car.s.t3, car.s.t4]
                    i += 1

                # get action
                # auto car -> net choose action
                # rancaom car -> expert rule choose action
                actions = []
                if self.epi_step < 10:
                    actions = [0, 0, 0, 1, 1, 1]
                else:
                    a = self.net.choose_actions(s[0], self.epsilon)
                    actions.append(a)
                    for j in range(1, len(self.car_list)):
                        a = self.choose_expert_actions(j)
                        actions.append(a)

                # step back
                s_, r, done = self.step(actions)
                # print('reward:', reward)

                # store memory
                # s:= observations
                # a:= actions
                # r:= rewards
                # s_:=observations_
                '''print('s:', s)
                print('a:', a)
                print('r:', r)
                print('s_:', s_)'''
                # for j in range(len(self.car_list)):
                #    self.net.store_transition(s[j], actions[j], r[j], s_[j])
                # 仅训练car0
                self.net.store_transition(s[0], actions[0], r[0], s_[0])

                # learn
                if self.net.memory_counter > MEMORY_SIZE:
                    self.net.learn()

                # draw
                self.clock = pygame.time.Clock()
                self.ticks = 60
                self.GUI.draw_window(2)
                for i in range(len(self.car_list)):
                    self.GUI.draw_car(self.car_list[i].position % 1600, self.car_list[i].lane - 1, i)
                pygame.display.flip()
                # delay = 60ms 刷新时间
                self.clock.tick(self.ticks)
                # sleep
                time.sleep(0.1)

                # is done
                if self.check_conflict() or self.epi_step >= self.max_step_in_every_episode:  # 发生碰撞或者达到最大运行步数
                    print('Is_conflict:', self.check_conflict())
                    print('Total_step_in_this_episode:', self.epi_step)
                    break

            if i_episode % 10 == 0:
                # update epsilon
                if i_episode > 100:
                    self.epsilon = 1
                else:
                    self.epsilon = 1 - (1 - self.epsilon) * 0.9

            print('Episode', self.episode, 'reward', self.r0 / self.epi_step)
            with open('../data/single_car_model_reward.csv', 'a', encoding='utf-8-sig', newline='' "") as file_handle:
                csv_writer = csv.writer(file_handle)
                csv_writer.writerow(
                    [self.episode, self.epi_step, self.r0 / self.epi_step])

        # save
        torch.save(self.net.eval_net.state_dict(), '../model/5_dim_eval_net_parameter_expert.pkl')
        torch.save(self.net.target_net.state_dict(), '../model/5_dim_target_net_parameter_expert.pkl')
        print("\n训练结束\n")

    def test_model(self):
        self.GUI = GUI()
        self.net_5_dim = DQN_5dim()
        self.net_5_dim.eval_net.load_state_dict(torch.load('../model/5_dim_eval_net_parameter_expert.pkl'))
        self.net_5_dim.target_net.load_state_dict(torch.load('../model/5_dim_target_net_parameter_expert.pkl'))

        for i_episode in range(100):
            # 新的episode
            self.episode += 1

            # reset and print_info
            self.reset()
            self.print_train_info()

            self.GUI.reset()

            # episode running
            while True:
                # step ++
                self.epi_step += 1

                # run step
                s = [[] for _ in range(len(self.car_list))]
                i = 0
                for car in self.car_list:
                    car.s = self.get_state(car.car_id)  # 获取每个车辆的状态信息
                    s[i] = [car.s.l, car.s.t1, car.s.t2, car.s.t3, car.s.t4]
                    i += 1

                # get action
                actions = []
                for j in range(len(self.car_list)):
                    a = self.net_5_dim.choose_actions(s[j], 1)
                    actions.append(a)
                # step back
                s_, r, done = self.step(actions)
                # print('reward:', reward)

                self.clock = pygame.time.Clock()
                self.ticks = 60

                self.GUI.draw_window(2)
                for i in range(len(self.car_list)):
                    self.GUI.draw_car(self.car_list[i].position % 1600, self.car_list[i].lane - 1)
                pygame.display.flip()

                # delay = 60ms 刷新时间
                self.clock.tick(self.ticks)
                # sleep
                time.sleep(0.1)

                # is done
                if self.check_conflict() or self.epi_step >= 400:  # 发生碰撞或者达到最大运行步数
                    print('Is_conflict:', self.check_conflict())
                    print('Total_step_in_this_episode:', self.epi_step)
                    break

            print('Episode', self.episode, 'reward', self.episode_reward / self.epi_step)


