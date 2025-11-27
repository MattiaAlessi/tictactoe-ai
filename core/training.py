import sys
import os
from termcolor import colored, cprint

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.neural_network import TicTacToeAI
from core.game_environment import TicTacToeGame
from utils.monitor import TrainingMonitor

def train_ai(episodes=1000, save_path="tictactoe_ai.pth"):
    ai = TicTacToeAI()
    monitor = TrainingMonitor()
    
    cprint("Starting AI training...", "cyan", attrs=['bold'])
    
    for episode in range(episodes):
        game = TicTacToeGame()
        state = game.reset()
        total_loss = 0
        steps = 0
        
        while not game.done:
            available_actions = game.get_available_actions()
            action = ai.choose_action(state, available_actions)
            next_state, reward, done = game.make_move(action)
            
            # SIMPLE REWARDS
            if done:
                if game.winner == 1:  # AI won
                    reward = 3
                elif game.winner == -1:  # AI lost
                    reward = -5
                else:  # Draw
                    reward = 0.1
            else:
                reward = 0  # No reward for continuing
            
            reward = max(min(reward, 5), -5)
            
            ai.remember(state, action, reward, next_state, done)
            state = next_state
            
            # Train less frequently
            if steps % 2 == 0:
                loss = ai.replay(batch_size=16)  # Smaller batch
                if loss:
                    total_loss += loss
                    steps += 1
            else:
                steps += 1
        
        # Update monitor
        win = 1 if game.winner == 1 else 0
        avg_loss = total_loss / (steps // 2) if (steps // 2) > 0 else 0
        monitor.update(win, avg_loss, ai.epsilon)
        
        if episode % 50 == 0:
            cprint(f"Episode {episode}, Epsilon: {ai.epsilon:.3f}, Loss: {avg_loss:.3f}", "yellow")
            
        # Stop if loss becomes unreasonable
        if avg_loss > 50:
            cprint(f"Stopping training - loss too high: {avg_loss:.3f}", "red")
            break
    
    ai.save_model(save_path)
    cprint(f"Training completed! Model saved as {save_path}", "green", attrs=['bold'])
    return ai, monitor

def continue_training(ai, additional_episodes=100):
    """Continue training an existing AI"""
    monitor = TrainingMonitor()
    
    cprint(f"Continuing training for {additional_episodes} episodes...", "blue")
    
    for episode in range(additional_episodes):
        game = TicTacToeGame()
        state = game.reset()
        total_loss = 0
        steps = 0
        
        while not game.done:
            available_actions = game.get_available_actions()
            action = ai.choose_action(state, available_actions)
            next_state, reward, done = game.make_move(action)
            
            # Simple rewards
            if done:
                if game.winner == 1:
                    reward = 3
                elif game.winner == -1:
                    reward = -5
                else:
                    reward = 0.1
            else:
                reward = 0
            
            reward = max(min(reward, 5), -5)
            
            ai.remember(state, action, reward, next_state, done)
            state = next_state
            
            if steps % 2 == 0:
                loss = ai.replay(batch_size=16)
                if loss:
                    total_loss += loss
                    steps += 1
            else:
                steps += 1
        
        # Update monitor
        win = 1 if game.winner == 1 else 0
        avg_loss = total_loss / (steps // 2) if (steps // 2) > 0 else 0
        monitor.update(win, avg_loss, ai.epsilon)
        
        if episode % 50 == 0:
            cprint(f"Additional Episode {episode}, Epsilon: {ai.epsilon:.3f}, Loss: {avg_loss:.3f}", "magenta")
            
        if avg_loss > 50:
            cprint(f"Stopping training - loss too high: {avg_loss:.3f}", "red")
            break
    
    return ai, monitor

if __name__ == "__main__":
    ai, monitor = train_ai(episodes=300)