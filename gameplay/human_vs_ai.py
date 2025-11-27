import sys
import os
from termcolor import colored, cprint

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.neural_network import TicTacToeAI
from core.game_environment import TicTacToeGame

def play_against_ai(ai, human_first=True, learn_from_game=True):
    """Play a game against the trained AI with optional learning"""
    game = TicTacToeGame()
    state = game.reset()
    
    cprint("Welcome Neural Network Tic-Tac-Toe", "cyan", attrs=['bold'])
    cprint("Cells are numbered as follows:", "white")
    cprint(" 0 | 1 | 2 ", "yellow")
    cprint("-----------", "white")
    cprint(" 3 | 4 | 5 ", "yellow") 
    cprint("-----------", "white")
    cprint(" 6 | 7 | 8 ", "yellow")
    print()
    
    human_player = 1 if human_first else -1
    ai_player = -human_player
    
    # Store game history for learning
    game_history = []
    
    while not game.done:
        print(game.display_board())
        
        if game.current_player == human_player:
            # Human turn
            try:
                action = int(input(colored("Enter your move (0-8): ", "green")))
                if action not in game.get_available_actions():
                    cprint("Invalid move", "red")
                    continue
            except ValueError:
                cprint("Please enter a valid number", "red")
                continue
        else:
            # AI turn
            available_actions = game.get_available_actions()
            action = ai.choose_action(state, available_actions)
            cprint(f"AI plays in cell {action}", "magenta")
        
        # Store move for learning
        old_state = state.copy()
        state, reward, done = game.make_move(action)
        game_history.append((old_state, action, reward, state, done))
    
    # Final result
    cprint("\n" + "="*30, "cyan")
    print(game.display_board())
    if game.winner == human_player:
        cprint("You won! Congratulations!", "green", attrs=['bold'])
    elif game.winner == ai_player:
        cprint("AI won! Try again!", "red", attrs=['bold'])
    else:
        cprint("Draw!", "yellow", attrs=['bold'])
    cprint("="*30, "cyan")
    
    # Learn from this game
    if learn_from_game:
        cprint("AI learning from this game...", "blue")
        for experience in game_history:
            ai.remember(*experience)
        # Train on accumulated experiences
        for _ in range(10):  # Mini-batch training
            ai.replay()
        cprint(f"AI updated! New epsilon: {ai.epsilon:.3f}", "green")
    
    return game.winner

def continue_training_during_play(ai, games_against_human=10):
    """Continue training AI while playing against human"""
    cprint("\nContinuous Learning Mode", "cyan", attrs=['bold'])
    cprint("AI will learn from your moves", "blue")
    
    for game_num in range(games_against_human):
        cprint(f"\n--- Game {game_num + 1} ---", "yellow", attrs=['bold'])
        play_against_ai(ai, human_first=(game_num % 2 == 0))
        
        # Continue training with mini-batch
        for _ in range(10):
            ai.replay()
        
        cprint(f"Current epsilon: {ai.epsilon:.3f}", "magenta")

def tournament_mode(ai, num_games=5):
    """Play multiple games in tournament mode"""
    human_wins = 0
    ai_wins = 0
    draws = 0
    
    cprint(f"\nTournament Mode - Best of {num_games} games", "cyan", attrs=['bold'])
    
    for game_num in range(num_games):
        cprint(f"\n--- Game {game_num + 1} ---", "yellow", attrs=['bold'])
        human_first = (game_num % 2 == 0)
        
        game = TicTacToeGame()
        state = game.reset()
        human_player = 1 if human_first else -1
        
        while not game.done:
            if game.current_player == human_player:
                print(game.display_board())
                try:
                    action = int(input(colored("Your move (0-8): ", "green")))
                    if action not in game.get_available_actions():
                        cprint("Invalid move", "red")
                        continue
                except ValueError:
                    cprint("Please enter a valid number", "red")
                    continue
            else:
                available_actions = game.get_available_actions()
                action = ai.choose_action(state, available_actions)
            
            state, _, _ = game.make_move(action)
        
        # Record result
        if game.winner == human_player:
            human_wins += 1
            cprint("You won this game", "green")
        elif game.winner == -human_player:
            ai_wins += 1
            cprint("AI won this game", "red")
        else:
            draws += 1
            cprint("Draw", "yellow")
    
    # Tournament results
    cprint(f"\n=== Tournament Results ===", "cyan", attrs=['bold'])
    cprint(f"Your wins: {human_wins}", "green")
    cprint(f"AI wins: {ai_wins}", "red")
    cprint(f"Draws: {draws}", "yellow")
    
    if human_wins > ai_wins:
        cprint("You won the tournament", "green", attrs=['bold'])
    elif ai_wins > human_wins:
        cprint("AI won the tournament", "red", attrs=['bold'])
    else:
        cprint("Tournament tied", "yellow", attrs=['bold'])