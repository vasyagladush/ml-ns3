#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from ns3gym import ns3env
from math import ceil

# simulation setup
port = 5556
simTime = 10  # seconds, first 2 seconds will be discarded
stepTime = 0.1  # seconds
# changing stepTime appears to require change both in .py and .cc
seed = 0
startSim = True
debug = False

if 2 >= simTime:
    raise ValueError("All simulation time will be spent on warm-up")
warmup_iterations = ceil(2 / stepTime)

simArgs = {"--simTime": simTime}

env = ns3env.Ns3Env(
    port=port,
    stepTime=stepTime,
    startSim=startSim,
    simSeed=seed,
    simArgs=simArgs,
    debug=debug,
)

# Q-learning parameters
alpha = 0.3
discount = 0.3
episodes = 25
disable_learning_after_episode = 20

action_count = 7  # [0,6]
state_collision_probability = 256  # uint8
state_packet_count = 256 # uint8
# No real reason to split the uint16 combined state here
shape = (state_collision_probability * state_packet_count, action_count)

Q = np.random.uniform(0.0, 1.0, size=shape).astype(np.float32)

rewards = []
iterations = []

for episode in range(episodes):
    learning = episode < disable_learning_after_episode
    if not learning:
        env.simSeed = 0

    try:
        raw = env.reset()
    except RuntimeError as e:
        print(
            f"Episode {episode}: could not reset environment (simulator stopped): {e}"
        )
        break

    state = raw  # TODO: divide by 'factor'
    t_reward = 0
    i = 0
    done = False

    for it in range(warmup_iterations):
        # ns3 seems to never really work before 2s mark of each episode
        # this will skip that time from Q-Learning's perspective
        env.step(0)

    while not done:
        i += 1
        s1 = state

        # choose action
        if learning:
            noise_scale = 1.0 / (episode + 1)
            tmpQ = Q[s1, :] + np.random.randn(action_count) * noise_scale
            action0 = int(np.argmax(tmpQ))
        else:
            action0 = int(np.argmax(Q[s1, :]))
        action = action0  # np.array([action0, 0], dtype=np.uint8)

        try:
            next_raw, reward, done, info = env.step(action)
        except RuntimeError as e:
            print(f"Episode {episode}, iter {i}: simulator stopped: {e}")
            done = True
            break

        t_reward += reward
        next_state = next_raw  # TODO: divide by 'factor'
        r1 = next_state

        if learning:
            best_next = np.max(Q[r1, :])
            Q[s1, action0] += alpha * (reward + discount * best_next - Q[s1, action0])

        state = next_state

    print(f"Episode {episode}, total reward: {t_reward}")
    rewards.append(t_reward)
    iterations.append(i)

env.close()

rewards = np.array(rewards)
chunks = np.array_split(rewards, episodes)
mean = np.mean(rewards[disable_learning_after_episode:])
averages = [sum(chunk) / len(chunk) for chunk in chunks]

print(f"Mean after learning: {mean}")
plt.plot(averages)
plt.xlabel("Episode")
plt.ylabel("Avg. reward")
plt.title("Average Reward (Episode)")
plt.savefig("avg.png")
plt.close()
