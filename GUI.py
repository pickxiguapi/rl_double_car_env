import pygame
import sys
from car import Car
import os
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d, %d" % (0, 100)

single_road_height = 110

class GUI(object):
    def __init__(self, roadway_num=2, ppu=16):
        # initial all
        pygame.init()
        self.roadway_num = roadway_num
        self.ppu = ppu

    def reset(self):
        # 设置窗口的分辨率和标题,绘制车道
        pygame.display.set_caption("Car tutorial")
        self.width = 800*2
        self.height = single_road_height*self.roadway_num
        self.screen = pygame.display.set_mode((self.width, self.height))

        '''background = pygame.image.load("road_1600x200_110.png").convert()
        for i in range(2):
            for j in range(self.roadway_num):
                self.screen.blit(background, (i*1600, j*single_road_height))'''

        # pygame.display.flip()

    def draw_car(self, position, lane):
        # 根据车的位置绘制车
        car_image = pygame.image.load("./pic/car2.png").convert_alpha()
        self.screen.blit(car_image, (position, lane*110+20))
        # print("draw_position:", position * self.ppu)

    def draw_window(self, roadway_num=2):
        # 绘制车道
        background = pygame.image.load("./pic/road_1600x200_110.png").convert()
        for i in range(1):
            for j in range(roadway_num):
                self.screen.blit(background, (i * 800, j * single_road_height))
        # car = pygame.image.load("car.png").convert_alpha()
        # self.screen.blit(car, (0, 0))

if __name__ == '__main__':
    road = GUI(2, 16)
    while True:
        # 处理退出事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        # 绘制背景和刷新界面
        road.draw_window(2)