from termcolor import colored, cprint
from core.neural_network import TicTacToeAI
from core.game_environment import TicTacToeGame

def analyze_ai_performance(ai, num_test_games=100):
    """Analyze AI performance"""
    wins_as_x = 0
    wins_as_o = 0
    draws = 0
    
    cprint(f"\nAnalyzing AI performance over {num_test_games} test games...", "cyan")
    
    # Test as X (first player)
    for i in range(num_test_games // 2):
        game = TicTacToeGame()
        state = game.reset()
        
        while not game.done:
            available_actions = game.get_available_actions()
            action = ai.choose_action(state, available_actions)
            state, _, _ = game.make_move(action)
        
        if game.winner == 1:  # X wins
            wins_as_x += 1
        elif game.winner == -1:  # O wins
            wins_as_o += 1
        else:
            draws += 1
    
    # Test as O (second player)
    for i in range(num_test_games // 2):
        game = TicTacToeGame()
        state = game.reset()
        
        # First move by random opponent
        available_actions = game.get_available_actions()
        import random
        random_action = random.choice(available_actions)
        state, _, _ = game.make_move(random_action)
        
        while not game.done:
            available_actions = game.get_available_actions()
            action = ai.choose_action(state, available_actions)
            state, _, _ = game.make_move(action)
        
        if game.winner == 1:  # X wins
            wins_as_x += 1
        elif game.winner == -1:  # O wins
            wins_as_o += 1
        else:
            draws += 1
    
    total_games = wins_as_x + wins_as_o + draws
    
    cprint(f"=== Performance Analysis ===", "cyan", attrs=['bold'])
    cprint(f"Wins as X: {wins_as_x}/{num_test_games}", "green")
    cprint(f"Wins as O: {wins_as_o}/{num_test_games}", "blue")
    cprint(f"Draws: {draws}/{num_test_games}", "yellow")
    cprint(f"Total win rate: {(wins_as_x + wins_as_o) / total_games * 100:.1f}%", "white")
    cprint(f"Win rate as X: {wins_as_x / (num_test_games) * 100:.1f}%", "green")
    cprint(f"Win rate as O: {wins_as_o / (num_test_games) * 100:.1f}%", "blue")
    
    return {
        'wins_as_x': wins_as_x,
        'wins_as_o': wins_as_o,
        'draws': draws,
        'total_win_rate': (wins_as_x + wins_as_o) / total_games
    }

def test_ai_strategy(ai, test_scenarios=None):
    """Test AI on specific board scenarios"""
    if test_scenarios is None:
        test_scenarios = [
            [1, 0, 0, 0, -1, 0, 0, 0, 1],  # Diagonal threat
            [1, -1, 0, 0, 0, 0, 0, 0, 0],  # Early game
            [1, -1, 1, -1, 0, 0, 0, 0, 0],  # Mid game
        ]
    
    cprint("\nTesting AI strategy on specific scenarios...", "cyan")
    
    for i, scenario in enumerate(test_scenarios):
        game = TicTacToeGame()
        game.board = scenario.copy()
        game.current_player = 1
        
        available_actions = game.get_available_actions()
        action = ai.choose_action(scenario, available_actions)
        
        cprint(f"Scenario {i+1}:", "yellow")
        cprint(f"Board: {scenario}", "white")
        cprint(f"AI chose action: {action}", "magenta")
        cprint(f"Available actions: {available_actions}", "blue")
        cprint("---", "white")