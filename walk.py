from env import RoomEnv
from model import DQN
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import time

env = RoomEnv()

# Environment parameters
n_actions = env.action_space.n
n_states = env.observation_space.shape[0]
print('n_actions', n_actions, 'n_states', n_states)
print(env.observation_space.sample())
exit()
# Hyper parameters
n_hidden = 4096
batch_size = 32
lr = 0.001                 # learning rate
epsilon = 0.1             # epsilon-greedy
gamma = 0.9               # reward discount factor
target_replace_iter = 100 # target network 更新間隔
memory_capacity = 100000
n_episodes = 10000

# 建立 DQN
dqn = DQN(n_states, n_actions, n_hidden, batch_size, lr, epsilon, gamma, target_replace_iter, memory_capacity)

# 學習
rewards_history = []
timestep_history = []
for i_episode in range(n_episodes):
    t = 0
    rewards = 0
    state = env.reset()
    while True:
        env.render()

        # 選擇 action
        action = dqn.choose_action(state)
        next_state, reward, done, info = env.step(action)
        # print(next_state)

        # 儲存 experience
        dqn.store_transition(state, action, reward, next_state)

        # 累積 reward
        rewards += reward

        # 有足夠 experience 後進行訓練
        if dqn.memory_counter > memory_capacity:
            dqn.learn()

        # 進入下一 state
        state = next_state

        if done:
            print('Episode {} finished after {} timesteps, total rewards {}'.format(i_episode, t+1, rewards))
            rewards_history.append(rewards)
            timestep_history.append(-t-1)
            # print('total count =', env.count)
            break

        t += 1
env.close()

# draw history value
title = 'Walking History'
total_steps = len(rewards_history)
x_1 = range(total_steps)
x_2 = x_1[::len(rewards_history) // len(rewards_history)]
figure(figsize=(6, 4))
plt.plot(x_1, rewards_history, c='tab:red', label='reward')
plt.plot(x_2, timestep_history, c='tab:cyan', label='timestep')
# plt.ylim(-10000, 0)
plt.xlabel('Training steps')
plt.ylabel('reward and timestep')
plt.title('Learning curve of {}'.format(title))
plt.legend()
plt.savefig('./reward_record.png')


input('pause now')

env = RoomEnv(render_mode='human')
# env = RoomEnv()

# Environment parameters
n_actions = env.action_space.n
n_states = env.observation_space.shape[0]
print('n_actions', n_actions, 'n_states', n_states)

n_episodes = 10


# 學習
rewards_history = []
timestep_history = []
for i_episode in range(n_episodes):
    print('Start Testing', i_episode)
    t = 0
    rewards = 0
    state = env.reset()
    while True:
        env.render()

        # 選擇 action
        action = dqn.choose_action(state, test=True)
        next_state, reward, done, info = env.step(action)
        # print(next_state)

        # 儲存 experience
        dqn.store_transition(state, action, reward, next_state)

        # 累積 reward
        rewards += reward

        # 有足夠 experience 後進行訓練
        if dqn.memory_counter > memory_capacity:
            dqn.learn()

        # 進入下一 state
        state = next_state

        if done:
            print('Episode {} finished after {} timesteps, total rewards {}'.format(i_episode, t+1, rewards))
            rewards_history.append(rewards)
            timestep_history.append(-t-1)
            # print('total count =', env.count)
            time.sleep(0.5)
            break

        t += 1

env.close()
