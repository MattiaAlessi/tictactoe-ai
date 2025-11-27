import matplotlib.pyplot as plt
import numpy as np

class TrainingMonitor:
    def __init__(self):
        self.wins = []
        self.losses = []
        self.epsilons = []
        self.win_rates = []
        
    def update(self, win, loss, epsilon):
        self.wins.append(win)
        self.losses.append(loss)
        self.epsilons.append(epsilon)
        
        # Calculate rolling win rate
        if len(self.wins) >= 100:
            recent_wins = self.wins[-100:]
            win_rate = sum(recent_wins) / len(recent_wins)
            self.win_rates.append(win_rate)
    
    def plot_progress(self, show_win_rate=True):
        plt.figure(figsize=(15, 5))
        
        # Plot wins
        plt.subplot(1, 3, 1)
        plt.plot(self.wins, alpha=0.3, label='Individual games')
        
        # Plot smoothed win rate
        if show_win_rate and len(self.win_rates) > 0:
            plt.plot(range(99, 99 + len(self.win_rates)), self.win_rates, 
                    color='red', linewidth=2, label='Win rate (last 100 games)')
        
        plt.title('Wins per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Wins')
        plt.legend()
        
        # Plot losses
        plt.subplot(1, 3, 2)
        plt.plot(self.losses)
        plt.title('Training Loss')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        
        # Plot epsilon
        plt.subplot(1, 3, 3)
        plt.plot(self.epsilons)
        plt.title('Exploration Rate (Epsilon)')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        
        plt.tight_layout()
        plt.show()
    
    def get_stats(self):
        """Get training statistics"""
        if len(self.wins) == 0:
            return {}
        
        stats = {
            'total_episodes': len(self.wins),
            'total_wins': sum(self.wins),
            'win_rate': sum(self.wins) / len(self.wins),
            'average_loss': np.mean(self.losses) if self.losses else 0,
            'current_epsilon': self.epsilons[-1] if self.epsilons else 1.0
        }
        
        if len(self.win_rates) > 0:
            stats['recent_win_rate'] = self.win_rates[-1]
        
        return stats
    
    def print_stats(self):
        """Print training statistics"""
        stats = self.get_stats()
        print("\n=== Training Statistics ===")
        for key, value in stats.items():
            print(f"{key.replace('_', ' ').title()}: {value:.3f}")