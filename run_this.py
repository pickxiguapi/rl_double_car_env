from env_10dim import Environment10
from six_car_env_10dim import Environment6

# 双车模型 10自主车 2车协同 8车使用单车模型的net
# env_10 = Environment10()
# train
# env_10.rl_method()
# test
# env_10.test_model()

env_6 = Environment6()
env_6.rl_method()
