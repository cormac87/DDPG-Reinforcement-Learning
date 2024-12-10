import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as D
from torch.distributions import Normal
from moviepy.editor import ImageSequenceClip

import random

class PolicyNN(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(PolicyNN, self).__init__()
        self.actions_mean = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_shape),
            nn.Tanh()
        )

    def forward(self, x):
        actions_mean = torch.clamp(self.actions_mean(x),-0.9,0.9)
        return actions_mean

class CriticNN(nn.Module):
    def __init__(self, obs_shape, act_shape):
        super(CriticNN, self).__init__()
        self.obs = nn.Sequential(
            nn.Linear(obs_shape, 64),
            nn.ReLU()
        )
        
        self.model = nn.Sequential(
            nn.Linear(64 + act_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, actions):
        obs_out = self.obs(x)
        return self.model(torch.cat([obs_out,actions],dim=1))

class Memory:
    def __init__(self, batch_size, observation_space, action_space):
        self.states_store = np.zeros((batch_size, observation_space))
        self.next_states_store = np.zeros((batch_size, observation_space))
        self.actions_store = np.zeros((batch_size, action_space))
        self.dones_store = np.zeros((batch_size, 1))
        self.rewards_store = np.zeros((batch_size, 1))
        
    def add_to_memory(self, batch_num, state, next_state, actions, done, reward):
        self.states_store[batch_num] = state
        self.next_states_store[batch_num] = next_state
        self.actions_store[batch_num] = actions
        self.dones_store[batch_num] = done
        self.rewards_store[batch_num] = reward

    def get_memory(self, minibatch):
        return (self.states_store[minibatch], self.next_states_store[minibatch], self.actions_store[minibatch], 
                self.dones_store[minibatch], self.rewards_store[minibatch])

    def sample_buffer(self, minibatch_size, buffer_size):
        minibatch = np.zeros(minibatch_size, dtype=int)
        for i in range(minibatch_size):
            minibatch[i] = np.random.randint(0,buffer_size)
        return self.get_memory(minibatch)
            
observation_space = 24
action_space = 4


actor_model = PolicyNN(observation_space, action_space)
target_actor_model = PolicyNN(observation_space, action_space)
critic_model = CriticNN(observation_space, action_space)
target_critic_model = CriticNN(observation_space, action_space)

critic_optimizer = optim.Adam(critic_model.parameters(), lr=0.001)
actor_optimizer = optim.Adam(actor_model.parameters(), lr=0.0001)

target_actor_model.load_state_dict(actor_model.state_dict())
target_critic_model.load_state_dict(critic_model.state_dict())

max_buffer_size = 1000000
buffer_size = 1000000

minibatch_size = 64

gamma = 0.99
r = 0.001
memory = Memory(max_buffer_size, observation_space, action_space)
env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
env = gym.wrappers.TransformReward(env, lambda rew: np.clip(rew, -10, 10))


def update_models(minibatch):
    (states, next_states, actions, dones, rewards) = minibatch
    with torch.no_grad():
        target_a = target_actor_model(torch.tensor(next_states,dtype=torch.float32))
        target_q = target_critic_model(torch.tensor(next_states,dtype=torch.float32),target_a)
    targets = torch.tensor(rewards,dtype=torch.float32) + gamma * target_q * (1-dones)
    critic_losses = torch.square(targets - critic_model(torch.tensor(states,dtype=torch.float32),torch.tensor(actions,dtype=torch.float32)))
    critic_loss = critic_losses.mean()
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    actor_losses = critic_model(torch.tensor(states,dtype=torch.float32), actor_model(torch.tensor(states,dtype=torch.float32)))
    actor_loss = -actor_losses.mean()
    critic_optimizer.zero_grad()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()


    with torch.no_grad():
        for target_param, new_param in zip(target_actor_model.parameters(), actor_model.parameters()):
            target_param.mul_(1-r).add_(r * new_param)
        for target_param, new_param in zip(target_critic_model.parameters(), critic_model.parameters()):
            target_param.mul_(1-r).add_(r * new_param)

                                 
noise_std = 0.2
mem_index = 0
frames = []
record_episode = False
do_training = False
tracked_rewards = 0
ep_count = 0
while True:
    ep_count += 1
    state = env.reset()[0]
    done = False
    if ep_count % 100 == 0:
        record_episode = True
        torch.save(actor_model.state_dict(), f'bipedal_model_{ep_count}.pth')
        frames = []

    while not done:
        with torch.no_grad():
            noise = torch.FloatTensor(1).uniform_(-noise_std, noise_std)
            actions = actor_model(torch.tensor(state,dtype=torch.float32))
        act = actions + noise
        next_state, reward, done, _, _ = env.step(act.numpy())
        memory.add_to_memory(mem_index, state, next_state, act, done, reward)
        state = next_state
        tracked_rewards += reward
        mem_index += 1
        
        if mem_index == max_buffer_size:
            mem_index = 0
            do_training = True
            buffer_size = max_buffer_size
        if not do_training:
            buffer_size = mem_index
        if mem_index % 10000 == 0:
            print(tracked_rewards / 10000)
            tracked_rewards = 0
        if record_episode:
            frame = env.render()
            frames.append(frame)
        minibatch = memory.sample_buffer(minibatch_size, buffer_size)
        update_models(minibatch)
    if record_episode:
        clip = ImageSequenceClip(frames, fps=30)
        clip.write_videofile(f'bipedalwalker_episode_{ep_count}.mp4', codec='mpeg4')
        record_episode = False
