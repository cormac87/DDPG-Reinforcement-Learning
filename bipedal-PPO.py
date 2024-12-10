import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as D
from torch.distributions import Normal

import random

env = gym.make("BipedalWalker-v3", render_mode="rgb_array")

class CriticModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(CriticModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fchidden = nn.Linear(128,128)
        self.fc2 = nn.Linear(128, output_size)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fchidden(x))
        x = self.fc2(x)
        return x

class ActorModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ActorModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fchidden = nn.Linear(128,128)
        self.fc2 = nn.Linear(128, output_size)
        self.log_1 = nn.Linear(input_size, 16)
        self.log_2 = nn.Linear(16, output_size)
    def forward(self, inp):
        x = F.relu(self.fc1(inp))
        x = F.relu(self.fchidden(x))
        means = torch.tanh(self.fc2(x))
        #y = torch.relu(self.log_1(inp))
        #stds = torch.sigmoid(self.log_2(y)) * 2
        #torch.clamp(stds, 0.1, 1.5)
        return means, torch.tensor(np.array([0.5,0.5,0.5,0.5]), dtype=torch.float32)




critic_input_size = 24
critic_output_size = 1

actor_input_size = 24
actor_output_size = 4

actor_model = ActorModel(actor_input_size, actor_output_size)
old_actor_model = ActorModel(actor_input_size, actor_output_size)

critic_model = CriticModel(critic_input_size, critic_output_size)

critic_optimizer = optim.Adam(critic_model.parameters(), lr=0.0003)
actor_optimizer = optim.Adam(actor_model.parameters(), lr = 0.0001)

criterion = nn.MSELoss() 


actor_model.train()
critic_model.train()
ep_count = 0
gamma = 0.99
alpha = 0.0003
lamda = 0.5
avg_reward = 0
old_actor_model.load_state_dict(actor_model.state_dict())
STATES_BUFFER = []
while True:
    states_actions_rewards = []
    count = 0
    done = False
    env.reset()
    x = env.step([0,0,0,0])
    state = x[0]
    ep_count += 1
    eligibility_trace = [torch.zeros_like(p) for p in critic_model.parameters()]
    height = 1
    while not done:
        count += 1
        critic_model.eval()
        with torch.no_grad():
            v_value = critic_model(torch.tensor(np.array(state),dtype=torch.float32))
        means, stds = actor_model(torch.tensor(np.array(state),dtype=torch.float32))
        actions = []
        for i in range(len(means)):
            mean = means[i]
            dist = Normal(mean,stds[i])
            n = dist.sample().item()
            if n < -1:
                n = -1
            if n > 1:
                n = 1
            actions.append(n)
        x = env.step(actions)
        next_state = x[0]
        height += next_state[3]
        reward = x[1]
        with torch.no_grad():
            next_v_value = critic_model(torch.tensor(np.array(next_state),dtype=torch.float32))
        critic_model.train()
        critic_optimizer.zero_grad()
        new_v_value = critic_model(torch.tensor(np.array(state),dtype=torch.float32))
        done = x[2]
        avg_reward += reward

        td_error = abs(reward + gamma * next_v_value * int(1 - done) - new_v_value)
        td_error.backward()
        torch.nn.utils.clip_grad_value_(critic_model.parameters(), 0.5)

        for param, trace in zip(critic_model.parameters(), eligibility_trace):
            trace.mul_(lamda * gamma).add_(param.grad)

        with torch.no_grad():
            for param, trace in zip(critic_model.parameters(), eligibility_trace):
                param.add_(alpha * td_error * -trace)

        states_actions_rewards.append((state, next_state, reward, actions))
        state = next_state
        if count > 1000 or done:
            done = True
            if ep_count % 50 == 0:
                avg_reward = avg_reward / 10000
                print(avg_reward, stds)
                avg_reward = 0
                torch.save(actor_model.state_dict(), 'bipedal_model.pth')

    for i in states_actions_rewards:
        STATES_BUFFER.append(i)

    if len(STATES_BUFFER) > 50000:
        over = len(STATES_BUFFER) - 50000
        STATES_BUFFER = STATES_BUFFER[over:]

    sample_size = 200
    if len(STATES_BUFFER) < 200:
        sample_size = len(STATES_BUFFER)
    random_sample = random.sample(STATES_BUFFER, sample_size)

    for z in range(3):
        total_actor_loss = 0
        for state, next_state, reward, actions in states_actions_rewards:
            with torch.no_grad():
                v_value = critic_model(torch.tensor(np.array(state),dtype=torch.float32))
                next_v_value = critic_model(torch.tensor(np.array(next_state),dtype=torch.float32))
                
            advantage = -(reward + gamma * next_v_value - v_value)

            means, stds = actor_model(torch.tensor(np.array(state),dtype=torch.float32))
            old_actor_model.eval()
            with torch.no_grad():
                old_means, old_stds = old_actor_model(torch.tensor(np.array(state),dtype=torch.float32))
            total_loss = 0
            for mean, old_mean, action, std, old_std in zip(means, old_means, actions, stds, old_stds):
                dist = Normal(mean,std)
                with torch.no_grad():
                    old_dist = Normal(old_mean, old_std)
                log_prob = dist.log_prob(torch.tensor(action,dtype=torch.float32))
                with torch.no_grad():
                    old_log_prob = old_dist.log_prob(torch.tensor(action,dtype=torch.float32))
                ratio = torch.exp(log_prob - old_log_prob)
                ratio_actor_loss = ratio * advantage
                clamped_ratio = torch.clamp(ratio, 0.8, 1.2)
                clipped_actor_loss = clamped_ratio * advantage
                if clipped_actor_loss < ratio_actor_loss:
                    actor_loss = clipped_actor_loss
                else:
                    actor_loss = ratio_actor_loss
                #entropy = 0.5 * (1 + torch.log(2 * torch.pi * std.pow(2)))
                total_loss += actor_loss + 0.1 * std #- entropy * 0.01

            total_actor_loss += total_loss

        total_actor_loss = total_actor_loss / len(states_actions_rewards)
        old_actor_model.load_state_dict(actor_model.state_dict())
        actor_optimizer.zero_grad()
        total_actor_loss.backward()
        torch.nn.utils.clip_grad_value_(actor_model.parameters(), 0.5)

        actor_optimizer.step()

