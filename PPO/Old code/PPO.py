import torch 
import torch.nn as nn
from torch.distributions import MultivariateNormal
from network import FeedForwardNN
import numpy as np
import gymnasium as gym

class PPO:
    def __init__(self, env, **hyperparameters):

        # Initialize hyperparameters
        self._init_hyperparameters(hyperparameters)

        # Extract environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape

        # Initialize actor and critic networks
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)

        self.cov_var = torch.full(size = (self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var) # create covariance matrix

        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)



    def compute_rewards_to_go(self, batch_rewards):
        batch_rewards_to_go = []

        for rewards in reversed(batch_rewards):
            discounted_reward = 0

            for i in reversed(rewards):
                discounted_reward = i + (discounted_reward * self.gamma)
                batch_rewards_to_go.insert(0, discounted_reward)
        batch_rewards_to_go = torch.tensor(batch_rewards_to_go, dtype=torch.float)

        return batch_rewards_to_go

    
    def _init_hyperparameters(self, hyperparameters):
        self.timesteps_per_batch = 4800                # timesteps per batch
        self.max_timesteps_per_episode = 1600          # timesteps per episode
        self.gamma = 0.95
        self.n_updates_per_iteration = 5
        self.clip = 0.2
        self.learning_rate = 0.005

    def get_action(self, observation):

        
        observation_tensor = torch.tensor(observation, dtype=torch.float)
        
        mean = self.actor(observation_tensor)

        dist = MultivariateNormal(mean, self.cov_mat)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach()

    def rollout(self):

        #Data 
        batch_observations = []
        batch_actions = []
        batch_log_probs = []
        batch_rewards = []
        batch_rewards_to_go = []
        batch_lens = []

        # Number of timesteps runs so far this batch 
        t = 0

        while t < self.timesteps_per_batch:

            # Rewards for this episode
            eps_rewards = []
            obs, _ = self.env.reset()
            done = False

            for ep in range(self.max_timesteps_per_episode):
                # Increment timesteps
                t += 1
                #print(t)
                # Collect observation
                batch_observations.append(obs)

                action, log_prob = self.get_action(obs)
                obs, reward, done , _, _ = self.env.step(action)

                # Collect reward, action and log prob
                eps_rewards.append(reward)
                batch_actions.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break

            # Collect episodic length and rewards
            batch_lens.append(ep+1)
            batch_rewards.append(eps_rewards)

            batch_observations = np.array(batch_observations)
            batch_observations = torch.tensor(batch_observations, dtype=torch.float)
            batch_actions = np.array(batch_actions)

            batch_actions = torch.tensor(batch_actions, dtype=torch.float)
            batch_log_probs = np.array(batch_log_probs)
            batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
            batch_rewards_to_go = self.compute_rewards_to_go(batch_rewards)

            return batch_observations, batch_actions, batch_log_probs, batch_rewards_to_go, batch_lens

        
    def evaluate(self, batch_obs, batch_actions):

        V = self.critic(batch_obs).squeeze()

        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_actions)

        return V, log_probs
    

    def learn(self, total_timesteps):
        current_step = 0 
        while current_step < total_timesteps:
            batch_obs, batch_action, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            current_step += np.sum(batch_lens)

            V = self.evaluate(batch_obs, batch_action)[0]
            #print(V)

            # Calculate the advantage
            Advantage = torch.tensor(batch_rtgs - V.detach(), requires_grad=True)
            #print(Advantage)

            # Normalize the advantage
            Advantage = (Advantage - Advantage.mean()) / (Advantage.std() + 1e-7) # To prevent division by zero
        
        for i in range(self.n_updates_per_iteration):
            _, current_log_probs = self.evaluate(batch_obs, batch_action)

            ratio = torch.exp(current_log_probs - batch_log_probs).detach()

            surrogate_loss1 = ratio * Advantage
            surrogate_loss2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * Advantage

            actor_loss = (-torch.min(surrogate_loss1, surrogate_loss2)).mean()
            critic_loss = nn.MSELoss()(V, batch_rtgs)

            # Back propogation
            self.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph = True)
            self.critic_optimizer.step

        print("Done!")


env = gym.make('highway-fast-v0', render_mode='rgb_array')
model = PPO(env)
model.learn(10000)