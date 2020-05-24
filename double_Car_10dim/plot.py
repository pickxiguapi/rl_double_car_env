import matplotlib.pyplot as plt
import csv
import numpy as np

episode_num = 3000
# epi_index, step, car0_reward, car5_reward, car_change_lane_number
epi, step, car0_reward, car5_reward, car_change_lane_number = [], [], [], [], []
with open('../data/double_car_model_reward.csv', encoding='utf-8') as f:
    reader = csv.reader(f)
    for i in range(episode_num):
        data = next(reader)
        epi.append(data[0])
        step.append(data[1])
        car0_reward.append(data[2])
        car5_reward.append(data[3])
        car_change_lane_number.append(data[4])
epi[0] = 1

epi = list(map(lambda x:float(x), epi))
step = list(map(lambda x: float(x), step))
car0_reward = list(map(lambda x: float(x), car0_reward))
car5_reward = list(map(lambda x: float(x), car5_reward))
car_change_lane_number = list(map(lambda x: float(x), car_change_lane_number))

plt.title('2 car + 4 expert car avg reward')
plt.xlabel('epi')
plt.ylabel('avg reward')
plt.plot(epi, car5_reward)
plt.savefig('../pic/2 car + 4 expert car avg reward.jpg')
plt.show()

Qlist = []
with open('../data/Q10dim.txt') as Q10:
    for q in Q10:
        q.rstrip('\n')
        Qlist.append(float(q))

qq = []
for i in range(len(Qlist)-100):
    qq.append(np.mean(Qlist[i:i+100]))

x = np.linspace(1, len(qq),len(qq))
plt.title('2 car + 4 expert car Q value')
plt.xlabel('step')
plt.ylabel('Q value')
plt.plot(x, qq)
plt.savefig('../pic/2 car + 4 expert car Q value.jpg')
plt.show()



