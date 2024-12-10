import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as D

env = gym.make("CartPole-v1", render_mode="rgb_array")



class ActorModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ActorModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fchidden = nn.Linear(16,16)
        self.fc2 = nn.Linear(16, output_size)
        self.sm = nn.Softmax(dim=1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fchidden(x))
        x = self.fc2(x)
        x = self.sm(x)
        return x

class CriticModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(CriticModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fchidden = nn.Linear(16,16)
        self.fc2 = nn.Linear(16, output_size)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fchidden(x))
        x = self.fc2(x)
        return x

input_size = 4
output_size = 2

model = ActorModel(input_size, output_size)

critic = CriticModel(input_size, 1)

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)
critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)
model.train()
critic.train()

actions = [0,1]

alpha = 0.9

counts = []
total_count = 0
while True:
    total_count += 1
    count = 0
    penalty = 0
    env.reset()
    x = env.step(np.random.randint(2))
    done = x[2]
    prev_state = x[0]
    env.render()
    total_reward = 1
    state_output_action = []
    states = []
    actions = []
    rewards = []
    while not done:
        count += 1
        prev_out = model(torch.tensor(np.array([prev_state]),dtype=torch.float32))
        m = D.Categorical(prev_out)
        prev_action = m.sample().item()
        with torch.no_grad():
            prev_v = critic(torch.tensor(np.array([prev_state]),dtype=torch.float32))
        x = env.step(prev_action)
        done = x[2]
        reward = 0
        if done:
            reward = 0
        else:
            reward = 1
        next_state = x[0]
        with torch.no_grad():
            next_out = model(torch.tensor(np.array([next_state]),dtype=torch.float32))
        next_m = D.Categorical(next_out)
        next_action = next_m.sample().item()
        with torch.no_grad():
            next_v = critic(torch.tensor(np.array([next_state]),dtype=torch.float32))
        action_prob = prev_out[0][prev_action]
        td_error = 0.95 * next_v * int(1 - done) + reward - prev_v
        actor_objective = torch.log(action_prob) * td_error
        actor_loss = -actor_objective
        optimizer.zero_grad()
        actor_loss.backward()
        optimizer.step()

        
        prev_v2 = critic(torch.tensor(np.array([prev_state]),dtype=torch.float32, requires_grad=True))
        critic_loss = abs(0.95 * next_v * int(1 - done) + reward - prev_v2)
        #print("estimate = ", prev_v)
        #print("target = ", next_v * int(1 - done) + reward)
        critic_optimizer.zero_grad()

        critic_loss.backward()
        
        critic_optimizer.step()

        prev_state = next_state

    print(count)
            
        
