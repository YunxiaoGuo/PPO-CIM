import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from network.actor_critic import *
from torch.distributions import Categorical


class Agent(object):
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.action_space = env.action_space
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.continuous = self.args.continuous
        self.actor = continuous_actor(env) if self.continuous == True else discrete_actor(env)
        self.critic = critic(env)
        self.actor.apply(init_weights)
        self.critic.apply(init_weights)
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.lr_a = self.args.lr_a
        self.lr_c = self.args.lr_c
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(),self.lr_a)
        self.optimizer_critic= torch.optim.Adam(self.critic.parameters(),self.lr_c)
        self.algo = self.args.algorithm
        self.epoch = self.args.num_epoch
        self.gamma = self.args.gamma
        self.gae_lambda = self.args.gae_lambda
        self.alpha=self.args.cim_alpha
        self.states=[]
        self.actions=[]
        self.log_probs=[]
        self.rewards=[]
        self.dones=[]
        self.new_states=[]
        self.values=[]
        #counters
        self.actor_count=0
        self.critic_count=0

    def select_action(self, obs):
        if self.args.continuous == False:
            prob = self.actor(torch.FloatTensor(obs).to(self.device))
            dist = Categorical(prob)
        else:
            mu, sigma = self.actor(torch.FloatTensor(obs).to(self.device))
            dist = torch.distributions.Normal(mu, sigma)
        action=dist.sample()
        self.log_probs.append(dist.log_prob(action))
        self.actions.append(action.detach())
        self.states.append(torch.FloatTensor(obs).to(self.device))
        self.values.append(self.critic(torch.FloatTensor(obs).to(self.device)).detach())
        return action.item() if self.args.continuous == False else action.tolist()

    def clean_buffer(self):
        self.states=[]
        self.actions=[]
        self.log_probs=[]
        self.rewards=[]
        self.dones=[]
        self.new_states=[]
        self.values=[]

    #PPO with CIM
    def learn_cim(self):
        last_val=self.critic(torch.FloatTensor(self.new_states[-1]).to(self.device)).item()
        rewards = np.zeros_like(self.rewards)
        advantage = np.zeros_like(self.rewards)
        adv=0.
        for t in reversed(range(len(self.rewards))):
            if t==len(self.rewards)-1:
                rewards[t]=self.rewards[t]+self.gamma*(1-self.dones[t])*last_val
                delta = self.rewards[t]+self.gamma*(1-self.dones[t])*last_val - self.values[t].item()
            else:
                rewards[t]=self.rewards[t]+self.gamma*(1-self.dones[t])*rewards[t+1]
                delta=self.rewards[t]+self.gamma*(1-self.dones[t])*self.values[t+1].item()-self.values[t].item()
            adv=adv*self.gamma*self.gae_lambda*(1-self.dones[t])+delta
            advantage[t]=adv
        rewards = torch.FloatTensor(rewards).to(self.device)
        advantage = torch.FloatTensor(advantage).to(self.device)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)
        old_states = torch.squeeze(torch.stack(self.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.log_probs, dim=0)).detach().to(self.device)
        if self.args.continuous == False:
            pi_old = self.actor(old_states).view((-1, self.action_space.n))
        else:
            mu_old, sigma_old = self.actor(old_states)
            theta_old = torch.cat((mu_old.detach(),sigma_old.detach()),dim=1)
        state_value=self.critic(old_states).view(-1)
        for _ in range(self.epoch):
            if self.args.continuous == False:
                probs = self.actor(old_states)
                dist = Categorical(probs)
            else:
                mu, sigma = self.actor(old_states)
                theta = torch.cat((mu,sigma),dim=1)
                dist = torch.distributions.Normal(mu, sigma)
            log_probs=dist.log_prob(old_actions)
            ratios=torch.exp(log_probs-old_logprobs.detach())
            advantage = advantage.reshape((advantage.shape[0],1))
            loss1=torch.mean(ratios*advantage.detach())
            sigma = self.args.cim_sigma
            if self.args.continuous == False:
                delta_pi = probs - pi_old.detach()
                if self.args.algorithm == 'CIM-1':
                    CIM = torch.pow(delta_pi, 2) / 2  # first-order expand
                elif self.args.algorithm == 'CIM-2':
                    CIM_1 = torch.pow(delta_pi, 2) / 2
                    CIM = CIM - torch.pow(CIM_1, 2) / 2  # second-order expand
                else:
                    CIM_1 = torch.pow(delta_pi, 2) / 2
                    CIM = 1/(np.sqrt(2*np.pi)*sigma)*torch.exp(-CIM/(sigma**2)) #no approximation
                loss2 = torch.mean(CIM)
            else:
                delta_pi = (theta - theta_old.detach())
                if self.args.algorithm == 'CIM-1':
                    CIM = torch.pow(delta_pi, 2) / 2  # first-order expand
                elif self.args.algorithm == 'CIM-2':
                    CIM_1 = torch.pow(delta_pi, 2) / 2
                    CIM = 1/(np.sqrt(2*np.pi)*sigma)*(CIM_1-torch.pow(CIM_1,2)/2) #second-order expand
                else:
                    CIM_1 = torch.pow(delta_pi, 2) / 2
                    CIM = 1/(np.sqrt(2*np.pi)*sigma)*torch.exp(-CIM_1/(sigma**2)) #no approximation
                loss2 = torch.mean(CIM)
            actor_loss = - (loss1-self.alpha*loss2)
            self.actor_count+=1
            self.actor_loss=actor_loss
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()
        loss=F.smooth_l1_loss(rewards,state_value.view(-1))
        self.critic_loss=loss
        self.critic_count+=1
        self.optimizer_critic.zero_grad()
        loss.backward()
        self.optimizer_critic.step()
        self.clean_buffer()

    def store(self,ob, action, new_obs, reward, done):
        self.rewards.append(reward)
        self.dones.append(float(done))
        self.new_states.append(new_obs)
    def timeToLearn(self):
        return len(self.new_states) >= self.args.batch_size

    def save_data(self, reward_data, episode):
        data_dir = os.path.join(self.args.data_path, self.args.env_name)
        data_dir = os.path.join(data_dir, self.args.algorithm)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        np.save(os.path.join(data_dir, str(episode) + '_' + self.args.algorithm + '_' + self.args.env_name + '.npy'),
                reward_data)