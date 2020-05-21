# rl_double_car_env
10 cars highway env, includes two cooperative car and eight car used pretraining network

# How to run
run "run_this.py"

# How to use GUI
run "env.test_model"

# How to train
run "env.rl_method"

# Other information
"eval_preprogress_model.pkl" := pretraining network for DQN algorithm of single car  
"target_preprogress_model.pkl" := pretraining network for DQN algorithm of single car  
"Q10dim.txt" := every step Qvalue  
"double_car_model_reward.txt" := every episode reward, every line format: r0 r9 global reward  
