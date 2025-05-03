import numpy as np
import matplotlib.pyplot as plt
import ns3env

# simulation setup
port = 5556
simTime = 10        # seconds
stepTime = 0.005    # seconds
seed = 0
startSim = True
debug = False

simArgs = {
    "--simTime": simTime,
    "--testArg": 123,
    "--distance": 20
}

env = ns3env.Ns3Env(
    port=port,
    stepTime=stepTime,
    startSim=startSim,
    simSeed=seed,
    simArgs=simArgs,
    debug=debug
)

# Q-learning parameters
alpha = 0.2
discount = 0.6
episodes = 25
disable_learning_after_episode = 15

action_count = 7               # [0,6]
state_queue_size = 256         # uint8
factor = 100.0 / (state_queue_size - 1)
shape = (state_queue_size, action_count)

Q = np.random.uniform(0.0, 1.0, size=shape).astype(np.float16)

rewards = []
iterations = []

for episode in range(episodes):
    learning = (episode <= disable_learning_after_episode)
    if not learning:
        env.simSeed = 0

    raw = env.reset()
    state = np.array(raw, dtype=np.uint8)[1:-1] / factor
    t_reward = 0
    i = 0
    done = False

    while not done:
        i += 1
        s1 = int(state[0])

        if learning:
            noise_scale = 1.0 / (episode + 1)
            tmpQ = Q[s1, :] + np.random.randn(action_count) * noise_scale
            action0 = int(np.argmax(tmpQ))
        else:
            action0 = int(np.argmax(Q[s1, :]))

        action = np.array([action0, 0], dtype=np.uint8)

        next_raw, reward, done, info = env.step(action)
        t_reward += reward

        next_state = np.array(next_raw, dtype=np.uint8)[1:-1] / factor
        r1 = int(next_state[0])

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
plt.show()