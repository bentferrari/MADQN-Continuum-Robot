import random
import gym
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam

EPISODES = 1000

class DQNAgent2:
    def __init__(self, n_actions, n_features):
        self.n_actions = n_actions
        self.n_features = n_features
        self.memory_size = 10000
        self.batch_size = 100
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.replace_target_iter = 20
        self.model1 = self._build_evaluate_net()
        self.model2 = self._build_target_net()

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 5 + 7))


    def _build_evaluate_net(self):
        # 建立Evaluate network
        model1 = Sequential(name='evaluate_network')
        model1.add(Dense(24, input_dim=self.n_features, activation='relu', name='layer1'))
        model1.add(Dense(24, activation='relu', name='layer2'))
        model1.add(Dense(self.n_actions, activation='linear', name='layer3'))
        model1.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model1

    def _build_target_net(self):
        model2 = Sequential(name='target_network')
        model2.add(Dense(24, input_dim=self.n_features, activation='relu'))
        model2.add(Dense(24, activation='relu'))
        model2.add(Dense(self.n_actions, activation='linear'))
        model2.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model2

    def store_transition(self, state, action1, action2, reward, next_state, done, reset, observe, next_observe):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((state, action1, action2, reward, next_state, done, reset, observe, next_observe))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]

        if np.random.rand() <= self.epsilon:

            action = np.random.randint(0, self.n_actions)
        else:
            act_values = self.model1.predict(observation)
            action = np.argmax(act_values[0])
        return action  # returns action

    def target_replace_op(self):
        v1 = self.model1.get_weights()
        self.model2.set_weights(v1)
        print("params has changed")

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_replace_op()
            print('\ntarget_params_replaced\n')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        q_next = self.model2.predict(batch_memory[:, [5, 6]])
        q_eval = self.model1.predict(batch_memory[:, [0, 1]])

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        done_batch = batch_memory[:, 7]
        reward_batch = batch_memory[:, 4]


        q_target[batch_index, eval_act_index] = reward_batch + self.gamma * np.max(q_next, axis=1)
                #q_target[batch_index, 1-eval_act_index] = reward_batch[count] + self.gamma * np.max(q_next, axis=1)

            #利用TD error来更新evaluate_net的参数
        self.model1.fit(batch_memory[:, :self.n_features], q_target, epochs=1, verbose=0)

        self.learn_step_counter += 1
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min


    def load(self):
        model1 = load_model("E://research_assistant/continuum_robot_script/python/2DQNAgent/rl2model1.h5")
        model2 = load_model("E://research_assistant/continuum_robot_script/python/2DQNAgent/rl2model2.h5")
        return model1, model2

    def save(self):
        self.model1.save("E://research_assistant/continuum_robot_script/python/2DQNAgent/rl2model1.h5")
        self.model2.save("E://research_assistant/continuum_robot_script/python/2DQNAgent/rl2model2.h5")
