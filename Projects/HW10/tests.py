import gymnasium as gym
import pickle
import random
import numpy as np
import time
import pdb

def default_Q_value():
    return 0


def evaluate_rl_agent(Q_table, EPSILON, env_name, visualize=False):
    total_reward = 0
    env = gym.envs.make(env_name)
    env.reset(seed=1)

    for i in range(100):
        obs = env.reset()[0]
        done = False
        while done == False:
            if random.uniform(0,1) < EPSILON:
                action = env.action_space.sample()
            else:
                prediction = np.array([Q_table[(obs,i)] for i in range(env.action_space.n)])
                action =  np.argmax(prediction)
            obs,reward,terminated,truncated,info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            if visualize:
                env.render()
                time.sleep(.01)
    score = total_reward/100
    return score



def test_RL_agent(config, visualize = False):
    env_name, algo_name = config[0], config[1]
    loaded_data = pickle.load(open(f'Q_TABLE_{algo_name}.pkl', 'rb'))
    Q_table = loaded_data[0]
    EPSILON = loaded_data[1]
    score = evaluate_rl_agent(Q_table,EPSILON,visualize = False, env_name = env_name)
    print(f"{algo_name} on {env_name}:")
    print("Average episode-reward over 100 episodes is " + str(score))
    # Gradescope will test on various thresholds for score.
    # For CliffWalking, you should aim for a score of >= -20 if you want to achieve full points.
    # For FrozenLake, you should aim for a score of >= 0.6



if __name__ == "__main__":
    print('-' * 40)
    config = ('CliffWalking-v0', 'QLearning')
    # config = ('CliffWalking-v0', 'SARSA')
    try:
        test_RL_agent(config)
    except Exception as e:
        print(e)