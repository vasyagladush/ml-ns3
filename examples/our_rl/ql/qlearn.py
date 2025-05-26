#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from ns3gym import ns3env
from math import ceil, exp, sin, pi
import random

# simulation setup
port = 5556
simTime = 12  # seconds, first 2 seconds will be discarded
stepTime = 0.01  # seconds
# changing stepTime appears to require change both in .py and .cc
seed = 0
startSim = True
debug = False

if 2 >= simTime:
    raise ValueError("All simulation time will be spent on warm-up")
warmup_iterations = ceil(2 / stepTime)
real_sim_time = (simTime - 2) / stepTime

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
discount = 0.2
episodes = 21
disable_learning_after_episode = 20
use_noise_x = True

if disable_learning_after_episode > episodes + 1:
    raise ValueError("Must have at least as many episodes as learning episodes + 1.")

action_count = 7  # [0,6]
state_collision_probability = 256  # uint8
state_packet_count = 256 # uint8
# No real reason to split the uint16 combined state here
shape = (state_collision_probability * state_packet_count, action_count)

#Q = np.random.uniform(0.0, 1.0, size=shape).astype(np.float32)
Q = np.zeros(shape, dtype=np.float32)

rewards = []
iterations = []

def get_noise(episode: int):
    episode = max(0, min(episode, disable_learning_after_episode - 1))
    x = episode / (disable_learning_after_episode - 1)
    noise = 0.6 * exp(-3 * x) + 0.2 * sin(3 * pi * x) + 0.3
    noise = max(0.1, min(noise, 1.0))
    return noise

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

    state = raw
    t_reward = 0
    i = 0
    done = False

    for it in range(warmup_iterations):
        # ns3 seems to never really work before 2s mark of each episode
        # this will skip that time from Q-Learning's perspective
        env.step(0)

    cws = []
    col_probs = []
    ths = []
    while not done:
        i += 1
        s1 = state

        col_prob = s1 & 255
        col_probs.append(col_prob)

        # choose action
        if learning:
            if use_noise_x:
                tmpQ = Q[s1, :] + np.random.randn(action_count) * get_noise(episode)
                action0 = int(np.argmax(tmpQ))
            else:
                noise_scale = 3.0 / (episode + 1)
                tmpQ = Q[s1, :] + np.random.randn(action_count) * noise_scale
                action0 = int(np.argmax(tmpQ))
        else:
            action0 = int(np.argmax(Q[s1, :]))
        action = action0  # np.array([action0, 0], dtype=np.uint8)
        #cws.append(2**(action + 3))
        cws.append(action)

        try:
            next_raw, reward, done, info = env.step(action)
        except RuntimeError as e:
            print(f"Episode {episode}, iter {i}: simulator stopped: {e}")
            done = True
            break

        t_reward += reward
        ths.append(reward)
        r1 = next_raw

        if learning:
            best_next = np.max(Q[r1, :])
            Q[s1, action0] += alpha * (reward + discount * best_next - Q[s1, action0])

        state = r1
    plt.plot(cws)
    plt.xlabel("iteration")
    plt.ylabel("CW")
    plt.title(f"CW (iteration), episode {episode}")
    plt.savefig(f"cw_ep{episode}")
    plt.close()

    plt.plot(col_probs)
    plt.xlabel("iteration")
    plt.ylabel("Collision Probability (x/255)")
    plt.title(f"Pr(Collision) (iteration), episode {episode}")
    plt.savefig(f"c_prob_ep{episode}")
    plt.close()

    plt.plot(ths)
    plt.xlabel("iteration")
    plt.ylabel("\"Throughput\"")
    plt.title(f"\"Throughput\" (iteration), episode {episode}")
    plt.savefig(f"ths_ep{episode}")
    plt.close()

    t_reward = t_reward / real_sim_time
    print(f"Episode {episode}, total reward: {t_reward}")
    rewards.append(t_reward)
    iterations.append(i)

env.close()

rewards = np.array(rewards)
chunks = np.array_split(rewards, episodes)
mean = np.mean(rewards[disable_learning_after_episode:])
averages = [sum(chunk) / len(chunk) for chunk in chunks]

# print(f"Mean after learning: {mean}")
plt.plot(averages)
plt.xlabel("Episode")
plt.ylabel("Avg. reward")
plt.title("Average Reward (Episode)")
plt.savefig("avg.png")
plt.close()
