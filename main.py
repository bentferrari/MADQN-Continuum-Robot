import gym
import continuum_gym
import matplotlib.pyplot as plt
import numpy as np
from RL1 import DQNAgent1
from RL2 import DQNAgent2
from matplotlib.animation import FuncAnimation


env = gym.make('continuum_env-v0')
ON_TRAIN = False


rl1 = DQNAgent1(n_actions=2, n_features=2)
rl2 = DQNAgent2(n_actions=2, n_features=2)


def train():

    total_steps = 0

    for i_episode in range(1, 100):

        state, observation = env.reset_code()
        ep_r = 0

        for itr_no in range(1, 500):
            # fresh env
            #env.render()

            # RL choose action based on observation
            action1Idx = rl1.choose_action(state)
            action2Idx = rl2.choose_action(state)

            # RL take action and get next observation and reward
            state, action1Idx, action2Idx, reward, next_state, done, reset, observation, next_observation, goal = env.step(action1Idx, action2Idx, observation)
            #print(['action2', action1Idx, 'action2', action2Idx, 'reward:', reward, 'state: ', state, 'is_done', done, 'goal', goal])

            rl1.store_transition(state, action1Idx, reward, next_state, done, reset, observation, next_observation)
            rl2.store_transition(state, action2Idx, action2Idx, reward, next_state, done, reset, observation, next_observation)

            ep_r += reward
            if total_steps > 10000:
                rl1.learn()
                rl2.learn()

            if done == 1:  # stop game
                break
            if reset == 1:
                break

            # swap observation
            state = next_state
            observation = next_observation
            total_steps += 1

        if i_episode % 10 == 0:
            rl1.save()
            rl2.save()

def evaluate():
    i_count = 0
    for j in range(100):
        if j == 0:
            state, observation = env.reset_code()
            rl1.load()
            rl2.load()
        else:
            state = env.reset_code1(state, i_count)
        for k in range(1, 50):
            action1Idx = rl1.choose_action(state)
            action2Idx = rl2.choose_action(state)
            state, action1Idx, action2Idx, reward, next_state, done, reset, observation, next_observation, goal = env.step(action1Idx, action2Idx, observation)
            rl1.store_transition(state, action1Idx, reward, next_state, done, reset, observation, next_observation)
            rl2.store_transition(state, action2Idx, action2Idx, reward, next_state, done, reset, observation, next_observation)
            rl1.learn()
            rl2.learn()
            state = next_state
            observation = next_observation
            #print(['itr_no', k, 'action1', action1Idx, 'action2', action2Idx, 'reward:', reward, 'state: ', state, 'goal', goal])
        rl1.save()
        rl2.save()
        i_count += 1

if ON_TRAIN:
    train()
else:
    evaluate()












