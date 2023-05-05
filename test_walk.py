import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import time

WALKING_PERIOD = 10000
model_path = 'dqn.ckpt'

class RoomEnv(Env):
    def __init__(self, width=21, height=21, render_mode=None):
        # Actions we can take, down, stay, up
        self.action_space = Discrete(4)
        # Temperature array
        self.observation_space = Box(low=-width//2, high=width//2, shape=(2,), dtype=np.int32)
        # Set start temp
        self.state = np.array([random.randint(0, width), random.randint(0, height)])
        # Set shower length
        self.walking_period = WALKING_PERIOD
        self.render_mode = render_mode

        self.width = width
        self.height = height

        
        
    def step(self, action):
        action_list = [
            np.array([1, 0]),
            np.array([0, 1]),
            np.array([-1, 0]),
            np.array([0, -1]),
        ]
        # Apply action
        self.state += action_list[action]
        if(self.state[0] < 0):
            self.state[0] = 0
        if(self.state[0] >= self.width):
            self.state[0] = self.width-1
        if(self.state[1] < 0):
            self.state[1] = 0
        if(self.state[1] >= self.height):
            self.state[1] = self.height-1
        
        # Reduce shower length by 1 second
        self.walking_period -= 1 
        
        # Calculate reward
        target = np.array([0, 0])
        reward = -(np.linalg.norm(self.state - target)) / 10
        reward -= 1
       
        # Check if shower is done
        if self.walking_period <= 0 or np.linalg.norm(self.state - target) == 0: 
            done = True
        else:
            done = False
        
        # Apply temperature noise
        #self.state += random.randint(-1,1)
        # Sblanket placeholder for info
        info = {}
        
        # Return step information
        return self.state, reward, done, info

    def render(self):
        if self.render_mode == None:
            return
        
        self.scene = [[' ']*self.height for _ in range(self.width)]
        self.scene[self.width//2+1][self.height//2+1] = 'T'
        self.scene[self.state[0]][self.state[1]] = 'A'

        print('============================================================')
        for i in range(self.width):
            for j in range(self.height):
                print(self.scene[i][j], end='')
            print()
        time.sleep(0.01)
        # if self.screen is None:
        #     pygame.init()
        #     if self.render_mode == 'human':
        #         pygame.display.init()
        #         self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))

        # if self.state is None:
        #     return None

        # self.surf = pygame.Surface((self.screen_width, self.screen_height))
        # self.surf.fill((120, 120, 120))
        # scale = self.screen_width / 10
        # radius = 20
        # gfxdraw.circle(self.surf, int(self.state[0]*scale), int(self.state[1]*scale), radius, (255, 0, 0))
        # print('Draw circle on', int(self.state[0]*scale), int(self.state[1]*scale))
        
        # self.surf = pygame.transform.flip(self.surf, False, True)
        # self.screen.blit(self.surf, (0, 0))
        
            
    
    def reset(self):
        # Reset shower temperature
        self.state = np.array([random.randint(0, self.width), random.randint(0, self.height)])
        # Reset shower time
        self.walking_period = WALKING_PERIOD
        return self.state
    
env = RoomEnv(render_mode='human')

# for i in range(10):
#     sample = env.observation_space.sample()
#     print(sample)

# episodes = 10
# for episode in range(1, episodes+1):
#     state = env.reset()
#     done = False
#     score = 0 
    
#     while not done:
#         #env.render()
#         action = env.action_space.sample()
#         n_state, reward, done, info = env.step(action)

#         score+=reward
#     print('Episode:{} Score:{}'.format(episode, score))

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden):
        super(Net, self).__init__()

        # 輸入層 (state) 到隱藏層，隱藏層到輸出層 (action)
        self.fc1 = nn.Linear(n_states, n_hidden)
        self.out = nn.Linear(n_hidden, n_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x) # ReLU activation
        actions_value = self.out(x)
        return actions_value

class DQN(object):
    def __init__(self, n_states, n_actions, n_hidden, batch_size, lr, epsilon, gamma, target_replace_iter, memory_capacity):
        self.eval_net, self.target_net = Net(n_states, n_actions, n_hidden), Net(n_states, n_actions, n_hidden)

        self.memory = np.zeros((memory_capacity, n_states * 2 + 2)) # 每個 memory 中的 experience 大小為 (state + next state + reward + action)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()
        self.memory_counter = 0
        self.learn_step_counter = 0 # 讓 target network 知道什麼時候要更新

        self.n_states = n_states
        self.n_actions = n_actions
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma
        self.target_replace_iter = target_replace_iter
        self.memory_capacity = memory_capacity
        
    def choose_action(self, state):
        x = torch.unsqueeze(torch.FloatTensor(state), 0)

        # epsilon-greedy
        if np.random.uniform() < self.epsilon: # 隨機
            action = np.random.randint(0, self.n_actions)
        else: # 根據現有 policy 做最好的選擇
            actions_value = self.eval_net(x) # 以現有 eval net 得出各個 action 的分數
            action = torch.max(actions_value, 1)[1].data.numpy()[0] # 挑選最高分的 action

        return action 
        
    def store_transition(self, state, action, reward, next_state):
        # 打包 experience
        transition = np.hstack((state, [action, reward], next_state))

        # 存進 memory；舊 memory 可能會被覆蓋
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # 隨機取樣 batch_size 個 experience
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        b_memory = self.memory[sample_index, :]
        b_state = torch.FloatTensor(b_memory[:, :self.n_states])
        b_action = torch.LongTensor(b_memory[:, self.n_states:self.n_states+1].astype(int))
        b_reward = torch.FloatTensor(b_memory[:, self.n_states+1:self.n_states+2])
        b_next_state = torch.FloatTensor(b_memory[:, -self.n_states:])

        # 計算現有 eval net 和 target net 得出 Q value 的落差
        q_eval = self.eval_net(b_state).gather(1, b_action) # 重新計算這些 experience 當下 eval net 所得出的 Q value
        q_next = self.target_net(b_next_state).detach() # detach 才不會訓練到 target net
        q_target = b_reward + self.gamma * q_next.max(1)[0].view(self.batch_size, 1) # 計算這些 experience 當下 target net 所得出的 Q value
        loss = self.loss_func(q_eval, q_target)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 每隔一段時間 (target_replace_iter), 更新 target net，即複製 eval net 到 target net
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

# Environment parameters
n_actions = env.action_space.n
n_states = env.observation_space.shape[0]
print('n_actions', n_actions, 'n_states', n_states)

# Hyper parameters
n_hidden = 50
batch_size = 32
lr = 0.001                 # learning rate
epsilon = 0.1             # epsilon-greedy
gamma = 0.9               # reward discount factor
target_replace_iter = 100 # target network 更新間隔
memory_capacity = 2000
n_episodes = 1000

# 建立 DQN
dqn = DQN(n_states, n_actions, n_hidden, batch_size, lr, epsilon, gamma, target_replace_iter, memory_capacity)
dqn.load_state_dict(torch.load(model_path))

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
            if t+1 < WALKING_PERIOD:
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
plt.ylim(-10000, 0)
plt.xlabel('Training steps')
plt.ylabel('reward and timestep')
plt.title('Learning curve of {}'.format(title))
plt.legend()
plt.savefig('./reward_record.png')
