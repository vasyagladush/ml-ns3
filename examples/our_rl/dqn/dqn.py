#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers
from ns3gym import ns3env
import matplotlib.pyplot as plt
from math import ceil

# --- DQN Agent Class ---
class DQNAgent:
    def __init__(self, state_size, action_size, cw_min=1, cw_max=100):
        self.state_size = state_size
        self.action_size = action_size
        self.cw_min = cw_min
        self.cw_max = cw_max

        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))  # dla CW każdego węzła
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.cw_min, self.cw_max+1, size=self.action_size)
        q_values = self.model.predict(np.array([state]), verbose=0)[0]
        return np.clip(np.round(q_values), self.cw_min, self.cw_max).astype(int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(np.array([state]), verbose=0)[0]
            if done:
                target = reward * np.ones(self.action_size)
            else:
                t = self.target_model.predict(np.array([next_state]), verbose=0)[0]
                target = reward + self.gamma * t
            self.model.fit(np.array([state]), np.array([target]), epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# --- NS3 Environment ---
port = 5555
simTime = 12
stepTime = 0.01
seed = 123
nodeNum = 5

if 2 >= simTime:
    raise ValueError("All simulation time will be spent on warm-up")
warmup_iterations = ceil(2 / stepTime)
real_sim_time = (simTime - 2) / stepTime

env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=True, simSeed=seed,
                    simArgs={"--simTime": simTime, "--nodeNum": nodeNum}, debug=False)

state_size = 2  # collision_prob i log2_queued_packets
action_size = env.action_space.shape[0]

agent = DQNAgent(state_size, action_size)
episodes = 50
episode_rewards = []
episode_cws = []
episode_throughputs = []
episode_collisions = []


for e in range(episodes):
    cws = []
    throughputs = []
    collisions = []

    raw_state = env.reset()
    collision_prob = raw_state[1] / 255.0         
    log2_queued = raw_state[0]           
    state = np.array([collision_prob, log2_queued])

    total_reward = 0

    for _ in range(warmup_iterations):
        env.step([7, 7, 7, 7, 7, 7])
    done = False

    while not done:
        action = agent.act(state)
        next_raw_state, reward, done, _ = env.step(action.tolist())

        collision_prob = next_raw_state[1] / 255.0  
        log2_queued = next_raw_state[0] 
        next_state = np.array([collision_prob, log2_queued])

        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        cws.append(np.mean(action)) 
        throughputs.append(reward)  
        collisions.append(next_raw_state[1] / 255.0) 

        if done:
            total_reward = total_reward / real_sim_time
            print(f"Episode {e+1}/{episodes} - reward: {total_reward}, epsilon: {agent.epsilon:.3f}")
            break

    episode_rewards.append(total_reward) 

    agent.replay()
    agent.update_target_model()
    episode_cws.append(np.mean(cws))
    episode_throughputs.append(np.mean(throughputs))
    episode_collisions.append(np.mean(collisions))

# --- Save reward plot ---


plt.figure(figsize=(10, 5))
plt.plot(episode_rewards, label='Total Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('DQN Training Progress')
plt.legend()
plt.grid(True)
plt.savefig("training_rewards.png")
# CW per episode
plt.figure(figsize=(10, 5))
plt.plot(episode_cws, label='Average CW')
plt.xlabel('Episode')
plt.ylabel('CW')
plt.title('Average Contention Window per Episode')
plt.grid(True)
plt.legend()
plt.savefig("avg_cw_per_episode.png")

# Throughput per episode
plt.figure(figsize=(10, 5))
plt.plot(episode_throughputs, label='Average Throughput', color='green')
plt.xlabel('Episode')
plt.ylabel('Throughput')
plt.title('Average Throughput per Episode')
plt.grid(True)
plt.legend()
plt.savefig("avg_throughput_per_episode.png")

# Collision probability per episode
plt.figure(figsize=(10, 5))
plt.plot(episode_collisions, label='Collision Probability', color='red')
plt.xlabel('Episode')
plt.ylabel('Collision Probability')
plt.title('Collision Probability per Episode')
plt.grid(True)
plt.legend()
plt.savefig("collision_prob_per_episode.png")

env.close()


