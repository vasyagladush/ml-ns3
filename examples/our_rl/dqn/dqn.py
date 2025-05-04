import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers
from ns3gym import ns3env

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
simTime = 10
stepTime = 0.1
seed = 123
nodeNum = 5

env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=True, simSeed=seed,
                    simArgs={"--simTime": simTime, "--nodeNum": nodeNum}, debug=False)

state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

agent = DQNAgent(state_size, action_size)
episodes = 50
max_steps = 100

for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [state_size])
    total_reward = 0

    for t in range(max_steps):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action.tolist())
        next_state = np.reshape(next_state, [state_size])

        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if done:
            print(f"Episode {e+1}/{episodes} - reward: {total_reward}, epsilon: {agent.epsilon:.3f}")
            break

    agent.replay()
    agent.update_target_model()

env.close()
