import random
import numpy as np
import torch
import torch.nn as nn
from util import state_to_array

class DummyModel:
    def predict(self, state, legal_actions):
        action = random.choice(legal_actions)
        action_prob = 1.0 / len(legal_actions)
        value = random.uniform(-1, 1)
        return action, action_prob, value

class SmallModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SmallModel, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(64, output_size)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = self.main(x)
        action_logits = self.action_head(x)
        value = self.value_head(x)
        value = torch.tanh(value)
        return action_logits, value

    def predict(self, state, legal_actions, deterministic=False):
        state = torch.tensor(state_to_array(state), dtype=torch.float32).unsqueeze(0)
        action_logit, value = self.forward(state)
        action_logit = action_logit.squeeze(0)
        value = value.squeeze(0)
        legal_action_indices = [a.value for a in legal_actions]
        legal_action_logit = action_logit[legal_action_indices]
        legal_action_probs = torch.softmax(legal_action_logit, dim=0)
        if deterministic:
            action_index = torch.argmax(legal_action_probs).item()
        else:
            action_index = torch.multinomial(legal_action_probs, 1).item()
        action = legal_actions[action_index]
        return action, legal_action_probs[action_index].item(), value.item()

    def reset(self):
        pass
    
    def load_path(self, path):
        weights = torch.load(path, map_location=torch.device('cpu'))
        self.load_weights(weights)
    def save_path(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, weights):
        self.load_state_dict(weights)
    def save_weights(self):
        return self.state_dict()
