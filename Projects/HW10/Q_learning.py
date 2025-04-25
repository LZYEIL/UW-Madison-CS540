import gymnasium as gym
import random
import numpy as np
import time
from collections import deque
import pickle


from collections import defaultdict


EPISODES =  30000
LEARNING_RATE = .1
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .999


def default_Q_value():
    return 0

if __name__ == "__main__":
    env_name = "CliffWalking-v0"
    env = gym.envs.make(env_name)
    env.reset(seed=1)

    # You will need to update the Q_table in your iteration
    Q_table = defaultdict(default_Q_value) # starts with a pessimistic estimate of zero reward for each state.
    episode_reward_record = deque(maxlen=100)

    for i in range(EPISODES):
        episode_reward = 0
        done = False
        obs = env.reset()[0]

        ##########################################################
        # YOU DO NOT NEED TO CHANGE ANYTHING ABOVE THIS LINE

        while (not done):
            
            # Epsilon-Greedy Behavior Policy:
            if (random.uniform(0, 1) > EPSILON):

                best_action = None
                best_q_value = float('-inf')
                
                # Get the action that maximizes the Q_value:
                for a in range(env.action_space.n):
                    q_value = Q_table[(obs, a)]

                    if q_value > best_q_value:
                        best_q_value = q_value
                        best_action = a

                action = best_action
            else:
                 action = env.action_space.sample() # performs a random action.

            current_state = obs  #The current state
            obs,reward,terminated,truncated,info = env.step(action)

            next_state = obs
            done = terminated or truncated

            episode_reward += reward

            current_q = Q_table[(current_state, action)]

            if not done:
                max_future_q = max([Q_table[(next_state, a)] for a in range(env.action_space.n)])
                Q_table[(current_state, action)] = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_future_q)
            else:
                Q_table[(current_state, action)] = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * reward    

            obs = next_state



        # YOU DO NOT NEED TO CHANGE ANYTHING BELOW THIS LINE
        ##########################################################

        # record the reward for this episode
        episode_reward_record.append(episode_reward) 
        EPSILON *= EPSILON_DECAY
    
        if i % 100 == 0 and i > 0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON) )
    
    
    #### DO NOT MODIFY ######
    model_file = open('Q_TABLE_QLearning.pkl', 'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    model_file.close()
    #########################