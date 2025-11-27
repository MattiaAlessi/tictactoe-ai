# 🧠 Neural Network Tic-Tac-Toe AI

[![Python](https://img.shields.io/badge/Python-3.8%252B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%252B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A sophisticated Tic-Tac-Toe AI that learns to play through Deep Q-Learning using PyTorch. Watch as the neural network improves its strategy through self-play and human interaction!

---

## 🎮 Features

- 🤖 **Adaptive AI** — Learns from every game it plays  
- 🎯 **Multiple Game Modes** — Casual play, continuous learning, tournaments  
- 📊 **Performance Analytics** — Track AI improvement with detailed statistics  
- 🌈 **Colorful Interface** — Beautiful terminal interface with emojis  
- 💾 **Save/Load Progress** — AI remembers what it learns between sessions  

---

## 🚀 Quick Start

### Installation

Clone the repository:

```bash
git clone https://github.com/MattiaAlessi/tictactoe-ai.git
cd tictactoe-ai
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the AI:

```bash
python main.py
```

---

## 🛠️ First Time Setup

On first run, the AI will train for 500 episodes against itself.  
There's an example

```yaml
Neural Network Tic-Tac-Toe AI
========================================
No existing model found. Training new AI...
Starting AI training...
Episode 0, Epsilon: 1.000, Loss: 0.000
Episode 100, Epsilon: 0.661, Loss: 0.074
Episode 200, Epsilon: 0.436, Loss: 0.111
Training completed! Model saved as tictactoe_ai.pth
```


**What's happening:**
- **Epsilon**: Exploration rate (higher = more random moves)
- **Loss**: How wrong the AI's predictions are (lower = better)
- The AI is learning basic Tic-Tac-Toe strategy through self-play

---

## 🎮 Game Modes

### 1. **Play Against AI** 
- Casual one-on-one matches
- Perfect for testing the AI's current skill level

### 2. **Continuous Learning Mode**
- Play multiple games in sequence  
- AI learns from each game and continues training
- Watch the AI improve in real-time

### 3. **Tournament Mode**
- Best-of-N series against the AI
- Pure competition (no learning during games)

### 4. **Train AI More**
- Intensive self-play training sessions
- Quickly boost the AI's skill level

---

## 🧠 How It Works

### Neural Network Architecture

```python
TicTacToeNet(
  (network): Sequential(
    (0): Linear(in_features=9, out_features=64, bias=True)
    (1): ReLU()
    (2): Linear(in_features=64, out_features=64, bias=True)
    (3): ReLU()
    (4): Linear(in_features=64, out_features=9, bias=True)
  )
)
```

**Reinforcement Learning**

- Deep Q-Learning algorithm

- Experience replay for stable training

- Epsilon-greedy exploration strategy

- Reward system: Win = +3, Loss = -5, Draw = +0.1

**Training Process**

- Exploration Phase — AI tries random moves (epsilon = 1.0)

- Exploitation Phase — AI uses learned strategies (epsilon decays to 0.3)

- Experience Replay — Learns from past games

- Continuous Improvement — Gets smarter with more gameplay

---

## 📁 Project Structure

```bash
tictactoe_ai/
├── core/
│   ├── neural_network.py    # AI brain 
│   ├── game_environment.py  # Tic-Tac-Toe rules 
│   └── training.py          # Learning algorithms 
├── gameplay/
│   └── human_vs_ai.py       # Player interaction 
├── utils/
│   ├── analysis.py          # Performance tracking 
│   └── monitor.py           # Training visualization 
├── main.py                  # Main application 
├── requirements.txt         
└── README.md                
```

---

## 📊 Performance Metrics

### The AI typically achieves:

- 85%+ win rate against itself

- Progressive improvement with more training

- Strategic diversity in opening moves

- Adaptive defense against human strategies

---

## ⚠️ Training Requirements for Best Performance

For the AI to reach strong and consistent performance, **extensive training is required**.  
While the model starts with basic self-play training, its real improvement comes from:

### 🏋️ 1. Large-Scale Training Sessions
The AI becomes significantly stronger when trained through:
- Thousands of self-play episodes  
- Repeated exploration and exploitation cycles  
- A well-filled experience replay memory  

More training = better decision making, fewer mistakes, and more strategic depth.

### 🎮 2. Playing Many Games Against Humans
Human opponents provide diverse, unpredictable strategies that self-play cannot fully replicate.  
The AI improves dramatically when it:
- Encounters different play styles  
- Learns human patterns and mistakes  
- Adapts to new tactics outside its self-play experience  

### 📈 Summary
To achieve optimal performance, ensure:
- **Extensive self-play training** (hundreds to thousands of episodes)  
- **Frequent games against human players** for real-world decision shaping  

The more the AI plays, the smarter it becomes.

---

## 🤝 Contributing

**Contributions are welcome! Here are some ways you can help:**

- Improve the neural network architecture

- Add new game modes

- Enhance the reward system

- Create a GUI interface

- Optimize training performance