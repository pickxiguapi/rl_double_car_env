from env import Environment
from env_10dim import Environment10
from env_1_auto_4_random import Environment_expert

# 单车模型 5自主车
# env = Environment_expert()
# env = Environment()
# train
# env.rl_method()
# test
# env.test_model()

# 双车模型 10自主车 2车协同 8车使用单车模型的net
env_10 = Environment10()
env_10.rl_method()