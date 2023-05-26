import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
import copy

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width, max_action,use_cnn = False):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.use_cnn = use_cnn
        if self.use_cnn:
            self.conv_1 = nn.Conv2d(1, 4, (3, 1))
            self.conv_2 = nn.Conv2d(1, 4, (3, 2))
            self.conv_3 = nn.Conv2d(1, 4, (3, 3))
            self.l1 = nn.Linear(108, hidden_width)
        else:
            self.l1 = nn.Linear(state_dim, hidden_width)

        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.mean_layer = nn.Linear(hidden_width, action_dim)
        self.log_std_layer = nn.Linear(hidden_width, action_dim)

    def forward(self, x, deterministic=False, with_logprob=True):
        state = x.to(torch.float32)
        if self.use_cnn:
            state = state.reshape((-1, 1, 3, int(self.state_dim / 3)))
            #TODO: This should be modified when your input dim is changed, 4 is the number of convolutional kernels
            out_1 = F.leaky_relu(torch.flatten(self.conv_1(state)).view(-1, 40))  #state_dim * 4
            out_2 = F.leaky_relu(torch.flatten(self.conv_2(state)).view(-1, 36))  #(state_dim-1) * 4
            out_3 = F.leaky_relu(torch.flatten(self.conv_3(state)).view(-1, 32))  #(state_dim-2) * 4
            out = torch.cat([out_1, out_2, out_3], dim=1)
            x = F.relu(self.l1(out))
        else:
            x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))

        # x = F.relu(self.l1(out))
        # x = F.relu(self.l2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)  # We output the log_std to ensure that std=exp(log_std)>0
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        dist = Normal(mean, std)  # Generate a Gaussian distribution
        if deterministic:  # When evaluating，we use the deterministic policy
            a = mean
        else:
            a = dist.rsample()  # reparameterization trick: mean+std*N(0,1)

        if with_logprob:  # The method refers to Open AI Spinning up, which is more stable.
            log_pi = dist.log_prob(a).sum(dim=1, keepdim=True)
            log_pi -= (2 * (np.log(2) - a - F.softplus(-2 * a))).sum(dim=1, keepdim=True)
        else:
            log_pi = None

        a = self.max_action * torch.tanh(
            a)  # Use tanh to compress the unbounded Gaussian distribution into a bounded action interval.

        return a, log_pi


class Critic(nn.Module):  # According to (s,a), directly calculate Q(s,a)
    def __init__(self, state_dim, action_dim, hidden_width,use_cnn = False):
        super(Critic, self).__init__()
        # Q1
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_cnn = use_cnn
        if self.use_cnn:
            self.conv_1 = nn.Conv2d(1, 4, (3, 1))
            self.conv_2 = nn.Conv2d(1, 4, (3, 2))
            self.conv_3 = nn.Conv2d(1, 4, (3, 3))
            # self.l1 = nn.Linear(input_dim + action_dim, hidden_width)
            self.l1 = nn.Linear(118, hidden_width)
        else:
            self.l1 = nn.Linear(state_dim + action_dim,hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, 1)

        if self.use_cnn:
            self.l4 = nn.Linear(118, hidden_width)
        else:
            self.l4 = nn.Linear(state_dim + action_dim, hidden_width)
        self.l5 = nn.Linear(hidden_width, hidden_width)
        self.l6 = nn.Linear(hidden_width, 1)

    def forward(self, s, a):
        state = s.to(torch.float32)
        if self.use_cnn:
            state = state.reshape((-1, 1, 3, int(self.state_dim / 3)))
            # TODO: This should be modified when your input dim is changed, 4 is the number of convolutional kernels
            out_1 = F.leaky_relu(torch.flatten(self.conv_1(state)).view(-1, 40))  #state_dim * 4
            out_2 = F.leaky_relu(torch.flatten(self.conv_2(state)).view(-1, 36))  #(state_dim-1) * 4
            out_3 = F.leaky_relu(torch.flatten(self.conv_3(state)).view(-1, 32))  #(state_dim-2) * 4
            s_a = torch.cat([out_1, out_2, out_3, a], dim=1)
        else:
            s_a = torch.cat([s, a], 1)
        q1 = F.relu(self.l1(s_a))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(s_a))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2

class SAC(object):
    def __init__(self, state_dim, action_dim, max_action,use_cnn):
        self.max_action = max_action
        self.hidden_width = 256  # The number of neurons in hidden layers of the neural network
        self.batch_size = 256  # batch size
        self.GAMMA = 0.99  # discount factor
        self.TAU = 0.005  # Softly update the target network
        self.lr = 2e-4  # learning rate
        # self.lr = 1e-4  # learning rate
        self.adaptive_alpha = True  # Whether to automatically learn the temperature alpha
        if self.adaptive_alpha:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.target_entropy = -action_dim
            # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)
        else:
            self.alpha = 0.2

        self.actor = Actor(state_dim, action_dim, self.hidden_width, max_action,use_cnn = use_cnn)
        self.critic = Critic(state_dim, action_dim, self.hidden_width,use_cnn=use_cnn)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    def choose_action(self, s, deterministic=False):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        a, _ = self.actor(s, deterministic, False)  # When choosing actions, we do not need to compute log_pi
        return a.data.numpy().flatten()

    def learn(self, relay_buffer):
        batch_s, batch_a, batch_r, batch_s_, batch_dw = relay_buffer.sample(self.batch_size)  # Sample a batch

        with torch.no_grad():
            batch_a_, log_pi_ = self.actor(batch_s_)  # a' from the current policy
            # Compute target Q
            target_Q1, target_Q2 = self.critic_target(batch_s_, batch_a_)
            target_Q = batch_r + self.GAMMA * (1 - batch_dw) * (torch.min(target_Q1, target_Q2) - self.alpha * log_pi_)

        # Compute current Q
        current_Q1, current_Q2 = self.critic(batch_s, batch_a)
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Freeze critic networks so you don't waste computational effort
        for params in self.critic.parameters():
            params.requires_grad = False

        # Compute actor loss
        a, log_pi = self.actor(batch_s)
        Q1, Q2 = self.critic(batch_s, a)
        Q = torch.min(Q1, Q2)
        actor_loss = (self.alpha * log_pi - Q).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critic networks
        for params in self.critic.parameters():
            params.requires_grad = True

        # Update alpha
        if self.adaptive_alpha:
            # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
            alpha_loss = -(self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        # Softly update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)