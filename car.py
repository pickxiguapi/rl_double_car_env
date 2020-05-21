import pygame
from state import State


class Car(pygame.sprite.Sprite):
    # 时间间隔
    delta_t = 1

    #  This class represents a car. It derives from the "Sprite" class in Pygame.
    def __init__(self, car_id, no, lane, velocity, position):
        pygame.sprite.Sprite.__init__(self)  # 继承自Sprite精灵类
        # 当前所在车道
        self.lane = lane
        # 车辆id
        self.car_id = car_id
        # 车辆编号
        self.no = no
        # 车辆位置 一维
        self.position = position
        # 车辆速度
        self.v0 = velocity
        # 当前状态
        self.s = State()
        # 下一状态
        self.s1 = State()
        # 当前选择动作，1为行车道，2为超车道
        self.a = 1
        # 加速度，初始化为0
        self.auto_a = 0.0
        # 行车道上的任务速度
        self.v_task = 30.0
        # 超车道上的超车速度
        self.v_overtake = 40.0
        # 规划速度
        self.v_plan = 0.0
        # 控制跟随速度
        self.v_bound = 2.0
        # 该车的累加奖励，初始为0
        self.r = 0.0

    def update_position(self):
        # 用的减号
        self.position += self.v0 * self.delta_t - 0.5 * self.auto_a * self.delta_t * self.delta_t

    def update_v0(self):
        self.v0 += self.auto_a

    def check_conflict(self, car_list):
        is_conflict = False
        for car in car_list:
            if car.no == self.no:
                continue
            if self.lane == car.lane and abs(car.position - self.position) <= 10:
                is_conflict = True
                break
            if self.lane == car.lane and car.position > self.position and car.position - car.v0 * self.delta_t + 0.5 * car.auto_a * self.delta_t * self.delta_t < self.position - self.v0 * self.delta_t + 0.5 * self.auto_a * self.delta_t * self.delta_t:
                is_conflict = True
                break
            if self.lane == car.lane and car.position < self.position and self.position - self.v0 * self.delta_t + 0.5 * self.auto_a * self.delta_t * self.delta_t < car.position - car.v0 * self.delta_t + 0.5 * car.auto_a * self.delta_t * self.delta_t:
                is_conflict = True
                break
        return is_conflict

    def print_info(self):
        print('-------------------- Begin --------------------')
        print('Car_id:',self.car_id)
        print('Lane:', self.lane)
        print('Position:', self.position)
        print('Velocity:', self.v0)
        print('Acc:', self.auto_a)
        print('Action:', self.a)
        print('--------------------  End  --------------------')

