from collections import deque
import random
import numpy as np

class TransitionBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=150000)
    def add(self, transition):
        self.buffer.append(transition)
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        indices = random.sample(range(len(self.buffer)), batch_size)
        return [self.buffer[i] for i in indices]
    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.buffer):
            raise IndexError("Index out of range")
        return self.buffer[idx]
    def __iter__(self):
        return iter(self.buffer)
    def __str__(self):
        return f'TransitionBuffer(size={len(self.buffer)})'
    def __repr__(self):
        return self.__str__()
