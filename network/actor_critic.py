import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    if isinstance(m,nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight,gain=torch.nn.init.calculate_gain('tanh'))
        torch.nn.init.zeros_(m.bias)

class continuous_actor(nn.Module):
    def __init__(self,env):
        super(continuous_actor, self).__init__()
        self.env=env
        self.action_space = env.action_space
        self.actor_fc1 = nn.Linear(env.observation_space.shape[0], 128)
        self.actor_fc2 = nn.Linear(128, 64)
        self.actor_fc3 = nn.Linear(64, self.action_space.shape[0])
        self.actor_fc4 = nn.Linear(64, self.action_space.shape[0])
        self.max_action = env.action_space.high[0]

    def forward(self,x):
        x = torch.tanh(self.actor_fc1(x))
        x = torch.tanh(self.actor_fc2(x))
        mean = torch.tanh(self.actor_fc3(x))
        std = torch.tanh(self.actor_fc4(x))
        mean = self.max_action*mean
        std=F.softplus(std)
        return mean,std

class discrete_actor(nn.Module):
    def __init__(self,env):
        super(discrete_actor, self).__init__()
        self.env=env
        self.action_space = env.action_space
        self.actor_fc1 = nn.Linear(env.observation_space.shape[0], 128)
        self.actor_fc2 = nn.Linear(128, 64)
        self.actor_fc3 = nn.Linear(64, self.action_space.n)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        x = torch.tanh(self.actor_fc1(x))
        x = torch.tanh(self.actor_fc2(x))
        x = self.actor_fc3(x)
        action_prob = self.softmax(x)
        return action_prob

class critic(nn.Module):
    def __init__(self,env):
        super(critic, self).__init__()
        self.env=env
        self.obs_space = self.env.observation_space.shape[0]
        self.critic_fc1 = nn.Linear(self.obs_space, 128)
        self.critic_fc2 = nn.Linear(128, 128)
        self.critic_fc3 = nn.Linear(128, 1)
    def forward(self,x):
        x = torch.tanh(self.critic_fc1(x))
        x = torch.tanh(self.critic_fc2(x))
        value = torch.tanh(self.critic_fc3(x))
        return value