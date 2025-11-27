import os
import sys
from termcolor import colored, cprint

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.training import train_ai, continue_training
from gameplay.human_vs_ai import play_against_ai, continue_training_during_play, tournament_mode
from utils.analysis import analyze_ai_performance
from utils.monitor import TrainingMonitor
from core.neural_network import TicTacToeAI

MODEL_PATH = "tictactoe_ai.pth"

def load_or_train_ai():
    """Load existing AI or train new one"""
    if os.path.exists(MODEL_PATH):
        cprint("Loading existing AI model...", "magenta")
        ai = TicTacToeAI()
        ai.load_model(MODEL_PATH)
        cprint(f"Model loaded! Current epsilon: {ai.epsilon:.3f}", "green")
        return ai
    else:
        cprint("No existing model found. Training new AI...", "yellow")
        ai, monitor = train_ai(episodes=500, save_path=MODEL_PATH) #first training = 500
        monitor.plot_progress()
        return ai

def main():
    cprint("Neural Network Tic-Tac-Toe", "cyan", attrs=['bold'])
    cprint("=" * 40, "cyan")
    
    # Load or train AI
    ai = load_or_train_ai()
    
    while True:
        cprint("\nMain Menu:", "white", attrs=['bold'])
        cprint("1. Play against AI", "yellow")
        cprint("2. Continuous learning mode", "blue") 
        cprint("3. Tournament mode", "green")
        cprint("4. Train AI more", "magenta")
        cprint("5. Analyze AI performance", "cyan")
        cprint("6. Show training progress", "white")
        cprint("7. Exit", "red")
        
        choice = input(colored("Choose an option (1-7): ", "white")).strip()
        
        if choice == "1":
            human_first = input(colored("Do you want to go first? (y/n): ", "yellow")).lower().strip() == 'y'
            play_against_ai(ai, human_first)
            
        elif choice == "2":
            games = int(input(colored("How many games to play? (default 3): ", "blue")) or 3)
            continue_training_during_play(ai, games)
            
        elif choice == "3":
            games = int(input(colored("How many tournament games? (default 3): ", "green")) or 3)
            tournament_mode(ai, games)
            
        elif choice == "4":
            episodes = int(input(colored("How many additional episodes? (default 100): ", "magenta")) or 100)
            ai, monitor = continue_training(ai, episodes)
            ai.save_model(MODEL_PATH)
            monitor.plot_progress()
            
        elif choice == "5":
            games = int(input(colored("How many test games? (default 50): ", "cyan")) or 50)
            analyze_ai_performance(ai, games)
            
        elif choice == "6":
            cprint("Training a new model to demonstrate progress...", "white")
            demo_ai, demo_monitor = train_ai(episodes=200)
            demo_monitor.plot_progress()
            
        elif choice == "7":
            ai.save_model(MODEL_PATH)
            cprint("Model saved. Thanks for playing", "green", attrs=['bold'])
            break
            
        else:
            cprint("Invalid choice", "red")

if __name__ == "__main__":
    main()