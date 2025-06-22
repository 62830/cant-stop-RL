import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from agents import RLAgent
from models import SmallModel
from game import RLGame
from transition_buffer import TransitionBuffer
from test import test
from util import state_to_array, STATE_LEN, ACTION_LEN, run_game

def get_legal_action_mask(state):
    legal_actions = state.get_legal_actions()
    legal_action_indices = [a.value for a in legal_actions]
    legal_action_mask = torch.tensor(
        [1 if i in legal_action_indices else 0 for i in range(12)],
        dtype=torch.float32
    )
    return legal_action_mask

def one_epoch(agent, buffer, optimizer, batch_size=64, num_steps=10, dev='cpu'):
    agent.model.train()
    
    for _ in tqdm(range(num_steps)):
        transitions = buffer.sample(batch_size)
        state, action, prob_old, reward, next_state, done = zip(*transitions)

        s_arr = torch.tensor(np.stack([state_to_array(s) for s in state]), dtype=torch.float32).to(dev)
        a_arr = torch.tensor([a.value for a in action], dtype=torch.long).to(dev)
        prob_old_arr = torch.tensor(prob_old, dtype=torch.float32).to(dev)
        r_arr = torch.tensor(reward, dtype=torch.float32).to(dev)
        ns_arr = torch.tensor(np.stack([state_to_array(s) for s in next_state]), dtype=torch.float32).to(dev)
        done_arr = torch.tensor(done, dtype=torch.float32).to(dev)
        legal_action_mask = torch.stack([get_legal_action_mask(s) for s in state]).to(dev)

        logit, value = agent.model(s_arr)
        with torch.no_grad():
            _, next_value = agent.model(ns_arr)
        
        # mask illegal actions in logit
        logit = logit + (1 - legal_action_mask) * -1e10
        prob = torch.softmax(logit, dim=1)
        logp = torch.log(prob.gather(1, a_arr.unsqueeze(1)).squeeze(1))
        logp_old = torch.log(prob_old_arr)
        ratio = torch.exp(logp - logp_old)

        # see if there are any NaN values in the ratio
        if torch.isnan(ratio).any():
            breakpoint()

        td_target = r_arr + (1 - done_arr) * next_value.squeeze(1)
        advantage = (td_target - value.squeeze(1)).detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 0.8, 1.2) * advantage
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = nn.functional.mse_loss(value.squeeze(1),td_target)
        loss = 100*policy_loss + value_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def train_rl_agent(weight_path=None, lr=3e-4, num_epoch=10, num_episodes=100, batch_size=64, num_steps=100, dev='cpu'):
    ppo_agent = RLAgent(SmallModel(STATE_LEN, ACTION_LEN))
    if weight_path:
        ppo_agent.model.load_path(weight_path)
        ppo_agent.model.to(dev)
    optimizer = optim.Adam(ppo_agent.model.parameters(), lr=lr)
    buffer = TransitionBuffer()
    rollout_agents = [RLAgent(SmallModel(STATE_LEN, ACTION_LEN), transition_buffer=buffer) for _ in range(2)]
    for rollout_agent in rollout_agents:
        rollout_agent.model.to(dev)
        rollout_agent.model.load_weights(ppo_agent.model.save_weights())

    for epoch in range(num_epoch):
        print(f'Epoch {epoch + 1}/{num_epoch}')
        print('Collecting episodes...')
        rollout_game = RLGame(rollout_agents)
        rollout_game.reset()
        buffer.clear()
        run_game(rollout_game, num_episodes=num_episodes, verbose=False)
        print(f'Buffer size: {len(buffer)}')
        print('Training...')
        one_epoch(ppo_agent, buffer, optimizer, batch_size=batch_size, num_steps=num_steps, dev=dev)

        for rollout_agent in rollout_agents:
            rollout_agent.model.load_weights(ppo_agent.model.save_weights())
        print('Testing...')
        test(rollout_agents[0])
        rollout_agents[0].model.save_path(f'model_weights/rl_agent_epoch_{epoch + 1}.pth')

        if epoch == 1:
            breakpoint()

if __name__ == '__main__':
    device = 'cpu'
    # weight_path = 'model_weights/rl_agent_epoch_30.pth'
    weight_path = None
    # train_rl_agent(weight_path=weight_path, lr=3e-4, num_epoch=10, num_episodes=50, batch_size=256, num_steps=150, dev=device)
    train_rl_agent(weight_path=weight_path, lr=3e-4, num_epoch=1000, num_episodes=500, batch_size=256, num_steps=1500, dev=device)