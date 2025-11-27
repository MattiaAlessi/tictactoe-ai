class TicTacToeGame:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.board = [0] * 9  # 0: empty, 1: X, -1: O
        self.current_player = 1  # X starts
        self.done = False
        self.winner = None
        return self.board.copy()
    
    def get_available_actions(self):
        return [i for i, cell in enumerate(self.board) if cell == 0]
    
    def make_move(self, action):
        """Execute a move"""
        if self.board[action] != 0 or self.done:
            return None, -10, True  # Illegal move
        
        self.board[action] = self.current_player
        
        # Check win
        if self.check_win(self.current_player):
            self.done = True
            self.winner = self.current_player
            return self.board.copy(), 10, True
        
        # Check draw
        if len(self.get_available_actions()) == 0:
            self.done = True
            return self.board.copy(), 0, True
        
        # Switch player
        self.current_player = -self.current_player
        return self.board.copy(), 1, False  # Small reward for valid move
    
    def check_win(self, player):
        """Check if a player has won"""
        board = self.board
        # Rows
        for i in range(0, 9, 3):
            if board[i] == board[i+1] == board[i+2] == player:
                return True
        # Columns
        for i in range(3):
            if board[i] == board[i+3] == board[i+6] == player:
                return True
        # Diagonals
        if board[0] == board[4] == board[8] == player:
            return True
        if board[2] == board[4] == board[6] == player:
            return True
        return False
    
    def display_board(self):
        """Display board in readable format"""
        symbols = {0: ' ', 1: 'X', -1: 'O'}
        board_str = ""
        for i in range(0, 9, 3):
            row = [symbols[self.board[i+j]] for j in range(3)]
            board_str += " " + " | ".join(row) + " \n"
            if i < 6:
                board_str += "-----------\n"
        return board_str
    
    def get_game_state(self):
        """Return current game state"""
        return {
            'board': self.board.copy(),
            'current_player': self.current_player,
            'done': self.done,
            'winner': self.winner,
            'available_moves': self.get_available_actions()
        }