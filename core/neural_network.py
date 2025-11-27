import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class TicTacToeNet(nn.Module):
    def __init__(self, input_size=9, hidden_size=32, output_size=9):  # Smaller network
        super(TicTacToeNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),  # More stable than ReLU
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class TicTacToeAI:
    def __init__(self, learning_rate=0.0001):  # Much smaller learning rate
        self.model = TicTacToeNet()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=2000)  # Smaller memory
        
        # Stable epsilon management
        self.epsilon = 1.0
        self.epsilon_min = 0.3  # Higher minimum
        self.epsilon_decay = 0.999
        self.gamma = 0.9
        
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss - more stable than MSE
        self.training_steps = 0
    
    def choose_action(self, state, available_actions):
        if np.random.random() <= self.epsilon:
            return random.choice(available_actions)
        else:
            state_tensor = self.get_state(state)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            
            available_q_values = {action: q_values[action].item() for action in available_actions}
            return max(available_q_values, key=available_q_values.get)
    
    def get_state(self, board):
        return torch.FloatTensor(board)
    
    def remember(self, state, action, reward, next_state, done):
        if action >= 0 and action < 9:
            self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return None
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([exp[0] for exp in batch])
        actions = torch.LongTensor([exp[1] for exp in batch])
        rewards = torch.FloatTensor([exp[2] for exp in batch])
        next_states = torch.FloatTensor([exp[3] for exp in batch])
        dones = torch.BoolTensor([exp[4] for exp in batch])
        
        # Current Q values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values - with target clamping
        with torch.no_grad():
            next_q_values = self.model(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
            
            # CRITICAL: Clamp targets to prevent explosion
            target_q_values = torch.clamp(target_q_values, -10, 10)
        
        # Calculate loss with Huber loss
        loss = self.loss_fn(current_q_values.squeeze(), target_q_values)
        
        # Update network with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        
        # CRITICAL: Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        
        self.training_steps += 1
        
        # Gentle epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def save_model(self, filepath):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps
        }, filepath)
    
    def load_model(self, filepath):
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint.get('training_steps', 0)