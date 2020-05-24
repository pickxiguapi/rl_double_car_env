import matplotlib.pyplot as plt
import csv
import numpy as np


'''
episode_num = 5000
# epi_index, step, car0_reward, car5_reward, car_change_lane_number
epi, step, car0_reward = [], [], []
with open('../data/single_car_model_reward.csv', encoding='utf-8') as f:
    reader = csv.reader(f)
    for i in range(episode_num):
        data = next(reader)
        epi.append(data[0])
        step.append(data[1])
        car0_reward.append(data[2])

epi[0] = 1

epi = list(map(lambda x:float(x), epi))
step = list(map(lambda x: float(x), step))
car0_reward = list(map(lambda x: float(x), car0_reward))

epi = epi[:3000]
car0_reward = car0_reward[:3000]
plt.title('1 car + 5 expert car avg reward')
plt.xlabel('epi')
plt.ylabel('avg reward')
plt.plot(epi, car0_reward)
plt.savefig('../pic/1 car + 5 expert car avg reward.jpg')
plt.show()'''

Qlist = []
with open('../data/Q5dim.txt') as Q5:
    for q in Q5:
        q.rstrip('\n')
        Qlist.append(float(q))

qq = []
print(len(Qlist))
for i in range(len(Qlist)-500000):
    qq.append(np.mean(Qlist[i:i+100]))


plt.title('1 car + 5 expert car Q value')
plt.xlabel('step')
plt.ylabel('Q value')
plt.plot(qq)
plt.savefig('../pic/1 car + 5 expert car Q value.jpg')
plt.show()



