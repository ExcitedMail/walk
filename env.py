import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import numpy as np
import random
import time


WALKING_PERIOD = 100

class RoomEnv(Env):
    def __init__(self, width=11, height=11, render_mode=None):
        # Actions we can take, down, stay, up
        self.action_space = Discrete(4)
        # Temperature array
        self.observation_space = Box(low=0, high=width-1, shape=(width, height), dtype=np.int32)
        # # Set start temp
        # self.state = np.array([random.randint(0, width-1), random.randint(0, height-1)])

        # self.target = np.array([random.randint(0, width-1), random.randint(0, height-1)])
        # Set shower length
        self.walking_period = WALKING_PERIOD
        self.render_mode = render_mode
        self.action = -1
        self.previous_state = (-1, -1)
        self.preprevious_state = (-1, -1)

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
        self.action = action
        bumpIntoWall = False
        # print('Action chosen is', self.action)
        self.state += action_list[action]
        if(self.state[0] < 0):
            self.state[0] = 0
            bumpIntoWall = True
        if(self.state[0] >= self.width):
            self.state[0] = self.width-1
            bumpIntoWall = True
        if(self.state[1] < 0):
            self.state[1] = 0
            bumpIntoWall = True
        if(self.state[1] >= self.height):
            self.state[1] = self.height-1
            bumpIntoWall = True
        
        
        # Reduce shower length by 1 second
        self.walking_period -= 1 
        
        # Calculate reward
        reward = -(np.linalg.norm(self.state - self.target))
        reward -= 1

        # if np.linalg.norm(self.state - self.preprevious_state) == 0:
        #     reward -= 10
        # self.preprevious_state = self.previous_state
        # self.previous_state = self.state

        if bumpIntoWall:
            reward -= 1000

        # Check if shower is done
        if np.linalg.norm(self.state - self.obstacle) == 0:
            reward -= 10000
            done = True
        elif np.linalg.norm(self.state - self.target) == 0:
            reward += 10000
            done = True
        elif self.walking_period <= 0:
            reward -= 10000
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
        
        # print('Position A now', self.state[0], self.state[1])
        
        self.scene = [[' ']*self.height for _ in range(self.width)]
        self.scene[self.target[0]][self.target[1]] = 'T'
        self.scene[self.state[0]][self.state[1]] = 'A'
        self.scene[self.obstacle[0]][self.obstacle[1]] = 'O'
        

        print('============================================================')
        for i in range(self.width):
            for j in range(self.height):
                print(self.scene[i][j], end='')
            print('|')
        time.sleep(0.05)
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
        
        self.state = np.array([0, 0])
        self.target = np.array([self.width-1, self.height-1])
        self.obstacle = np.array([self.width//2, self.height//2])
        
        # self.state = np.array([random.randint(0, self.width-1), random.randint(0, self.height-1)])
        # self.target = np.array([random.randint(0, self.width-1), random.randint(0, self.height-1)])
        # self.obstacle = np.array([random.randint(0, self.width-1), random.randint(0, self.height-1)])
        
        # Reset shower time
        self.walking_period = WALKING_PERIOD
        return self.state
  