import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from agents import RLAgent, MixedAgent, Agent
from fraction_eval import Fraction_Eval_Agent, Fraction_Eval_Agent_v2
from models import SmallModel, LargeModel, SepModel
from game import RLGame
from transition_buffer import TransitionBuffer
from test import test
from util import state_to_array, STATE_LEN, ACTION_LEN, run_game, run_game_parallel

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

    logging = {
        'policy_loss': [],
        'value_loss': [],
        'entropy': [],
        'clip_frac': []
    }
    
    for _ in tqdm(range(num_steps)):
        transitions = buffer.sample(batch_size)
        # state, action, prob_old, reward, next_state, done = zip(*transitions)
        state, action, prob_old, value_old, next_state, done, advantage = zip(*transitions)

        s_arr = torch.tensor(np.stack([state_to_array(s) for s in state]), dtype=torch.float32).to(dev)
        a_arr = torch.tensor([a.value for a in action], dtype=torch.long).to(dev)
        prob_old_arr = torch.tensor(prob_old, dtype=torch.float32).to(dev)
        value_old_arr = torch.tensor(value_old, dtype=torch.float32).to(dev)
        # ns_arr = torch.tensor(np.stack([state_to_array(s) for s in next_state]), dtype=torch.float32).to(dev)
        # done_arr = torch.tensor(done, dtype=torch.float32).to(dev)
        legal_action_mask = torch.stack([get_legal_action_mask(s) for s in state]).to(dev)
        advantage = torch.tensor(advantage, dtype=torch.float32).to(dev)
        advantage_normed = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        logit, value = agent.model(s_arr)
        # mask illegal actions in logit
        logit = logit + (1 - legal_action_mask) * -1e10
        prob = torch.softmax(logit, dim=1)
        logp = torch.log(prob.gather(1, a_arr.unsqueeze(1)).squeeze(1))
        logp_old = torch.log(prob_old_arr)
        ratio = torch.exp(logp - logp_old)

        surr1 = ratio * advantage_normed
        surr2 = torch.clamp(ratio, 0.8, 1.2) * advantage_normed
        policy_loss = -torch.min(surr1, surr2).mean() * 10

        value_target = advantage + value_old_arr
        value_loss = nn.functional.mse_loss(value.squeeze(), value_target)

        entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1).mean()
        clip_frac = ((ratio < 0.8) | (ratio > 1.2)).float().mean()
        loss = policy_loss + value_loss - 0.01 * entropy
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logging['policy_loss'].append(policy_loss.item())
        logging['value_loss'].append(value_loss.item())
        logging['entropy'].append(entropy.item())
        logging['clip_frac'].append(clip_frac.item())
    
    avg_policy_loss = np.mean(logging['policy_loss'])
    avg_value_loss = np.mean(logging['value_loss'])
    avg_entropy = np.mean(logging['entropy'])
    avg_clip_frac = np.mean(logging['clip_frac'])
    print(f'Policy Loss: {avg_policy_loss:.4f}, Value Loss: {avg_value_loss:.4f}, Entropy: {avg_entropy:.4f}, Clip Fraction: {avg_clip_frac:.4f}')

def train_rl_agent(weight_path=None, lr=3e-4, num_epoch=10, num_episodes=100, batch_size=64, num_steps=100, dev='cpu', version='v1', num_workers=8):
    model_cls = SepModel
    # model_cls = SmallModel
    # model_cls = LargeModel
    ppo_agent = RLAgent(model_cls(STATE_LEN, ACTION_LEN))
    if weight_path:
        ppo_agent.model.load_path(weight_path)
        ppo_agent.model.to(dev)
    optimizer = optim.Adam(ppo_agent.model.parameters(), lr=lr)
    buffer = TransitionBuffer()

    # rollout_agents hold the two self-play policies: agent 0 is the learner,
    # agent 1 is the frozen opponent. Rollouts run in worker processes
    # (run_game_parallel), so these objects are just weight holders + promotion
    # bookkeeping; only the learner's transitions are collected (on-policy).
    rollout_agents = [
        RLAgent(model_cls(STATE_LEN, ACTION_LEN)),
        RLAgent(model_cls(STATE_LEN, ACTION_LEN))
    ]
    # first agent updates quickly,
    # second agent updates only when first agent beats it more than 60% of the time

    for rollout_agent in rollout_agents:
        rollout_agent.load_weights(ppo_agent.model.save_weights())


    for epoch in range(num_epoch):
        print(f'Epoch {epoch + 1}/{num_epoch}')
        buffer.clear()
        wins, transitions = run_game_parallel(
            model_cls,
            rollout_agents[0].model.save_weights(),
            rollout_agents[1].model.save_weights(),
            num_episodes=num_episodes,
            num_workers=num_workers,
        )
        for transition in transitions:
            buffer.add(transition)
        print(f'Buffer size: {len(buffer)} Average episode length: {(len(buffer) / num_episodes):.2f} First agent wins: {wins[0]}, Second agent wins: {wins[1]}')
        if wins[0] >= 0.59 * num_episodes:
            print(f'Update second agent weights at epoch {epoch + 1}')
            rollout_agents[1].load_weights(rollout_agents[0].model.save_weights())
            rollout_agents[0].model.save_path(f'model_weights/rl_agent_{version}_{epoch + 1}.pth')
            test(rollout_agents[0], deterministic=True)

        one_epoch(ppo_agent, buffer, optimizer, batch_size=batch_size, num_steps=num_steps, dev=dev)

        rollout_agents[0].load_weights(ppo_agent.model.save_weights())

if __name__ == '__main__':
    device = 'cpu'
    # weight_path = 'model_weights/rl_agent_v8_677.pth'
    weight_path = None
    train_rl_agent(weight_path=weight_path, lr=3e-4, num_epoch=10000, num_episodes=400, batch_size=256, num_steps=300, dev=device,
                   version='v14', num_workers=12)
