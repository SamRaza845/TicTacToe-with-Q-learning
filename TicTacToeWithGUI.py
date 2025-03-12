import numpy as np
import tkinter as tk
import random
import pickle
import os

# Defining global variables
current_player = None
game_over = False
game_frame = None
buttons = None
remarks_label = None
restart_button = None
win_counts_label = None

human_wins = 0
ai_wins = 0
draws = 0

# Initialize the board
board = np.array([['-', '-', '-'],
                  ['-', '-', '-'],
                  ['-', '-', '-']])
players = ['X', 'O']
num_players = len(players)
Q = {}

learning_rate = 0.2  # Increased learning rate
discount_factor = 0.9
exploration_rate = 1.0  # Start with high exploration rate
exploration_decay = 0.999  # Gradually decrease exploration rate
min_exploration_rate = 0.01  # Minimum exploration rate
num_episodes = 200000  # Increased number of episodes
save_interval = 1000  # Save Q-table every 1000 episodes
q_table_file = 'TicTacToe.pkl'

random_count = 0
qcount = 0

# Function to convert the board state to a string to use it as a key in the Q-table dictionary.
def board_to_string(board):
    return ''.join(board.flatten())

def is_game_over(board):
    # Check rows for winning condition
    for row in board:
        if len(set(row)) == 1 and row[0] != '-':
            return True, row[0]

    # Check columns
    for col in board.T:
        if len(set(col)) == 1 and col[0] != '-':
            return True, col[0]

    # Check diagonals
    if len(set(board.diagonal())) == 1 and board[0, 0] != '-':
        return True, board[0, 0]
    if len(set(np.fliplr(board).diagonal())) == 1 and board[0, 2] != '-':
        return True, board[0, 2]

    # Check if the board is full
    if '-' not in board:
        return True, 'draw'

    return False, None

def get_winning_line(board):
    # Check rows
    for i, row in enumerate(board):
        if len(set(row)) == 1 and row[0] != '-':
            return ('row', i)

    # Check columns
    for i, col in enumerate(board.T):
        if len(set(col)) == 1 and col[0] != '-':
            return ('col', i)

    # Check main diagonal
    if len(set(board.diagonal())) == 1 and board[0, 0] != '-':
        return ('diag', 0)

    # Check anti-diagonal
    if len(set(np.fliplr(board).diagonal())) == 1 and board[0, 2] != '-':
        return ('diag', 1)

    return None

# Function to check for immediate win or block
def find_immediate_win_or_block(board, player):
    for i in range(3):
        for j in range(3):
            if board[i][j] == '-':
                # Try the move and see if it results in a win
                board[i][j] = player
                game_over, winner = is_game_over(board)
                board[i][j] = '-'
                if game_over and winner == player:
                    return (i, j)
    return None

# Updated function to choose an action based on Q-table and logical AI methods
def choose_action(board, exploration_rate):
    global random_count, qcount
    state = board_to_string(board)

    # Logical AI methods: Check for immediate wins or blocks
    win_move = find_immediate_win_or_block(board, 'O')
    if win_move:
        return win_move

    block_move = find_immediate_win_or_block(board, 'X')
    if block_move:
        return block_move

    # Take center if available
    if board[1][1] == '-':
        return (1, 1)

    # Take a corner if available
    corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
    empty_corners = [corner for corner in corners if board[corner[0], corner[1]] == '-']
    if empty_corners:
        return random.choice(empty_corners)

    # Exploration-exploitation trade-off
    if random.uniform(0, 1) < exploration_rate or state not in Q:
        # Choose a random action
        random_count += 1
        empty_cells = np.argwhere(board == '-')
        action = tuple(random.choice(empty_cells))
    else:
        qcount += 1
        # Choose the action with the highest Q-value
        q_values = Q[state]
        empty_cells = np.argwhere(board == '-')
        empty_q_values = [q_values[cell[0], cell[1]] for cell in empty_cells]
        max_q_value = max(empty_q_values)
        max_q_indices = [i for i in range(len(empty_cells)) if empty_q_values[i] == max_q_value]
        max_q_index = random.choice(max_q_indices)
        action = tuple(empty_cells[max_q_index])

    return action

# Convert the cell coordinates (row and column) of the chosen action to the next state of the board as a string.
def board_next_state(cell):
    next_state = board.copy()
    next_state[cell[0], cell[1]] = players[0]
    return next_state

def update_q_table(state, action, next_state, reward):
    q_values = Q.get(state, np.zeros((3, 3)))
    next_q_values = Q.get(board_to_string(next_state), np.zeros((3, 3)))
    max_next_q_value = np.max(next_q_values)
    q_values[action[0], action[1]] += learning_rate * (reward + discount_factor * max_next_q_value - q_values[action[0], action[1]])
    Q[state] = q_values

def save_q_table(file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(Q, file)

def load_q_table(file_name):
    global Q
    try:
        with open(file_name, 'rb') as file:
            Q = pickle.load(file)
    except FileNotFoundError:
        Q = {}

# Main Q-learning algorithm with logical AI methods
def train_q_table():
    global exploration_rate, random_count, qcount
    for episode in range(num_episodes):
        board = np.array([['-', '-', '-'],
                          ['-', '-', '-'],
                          ['-', '-', '-']])
        current_player = random.choice(players)
        game_over = False
        state_action_pairs = []

        while not game_over:
            if current_player == 'O':
                action = choose_action(board, exploration_rate)
            else:
                empty_cells = np.argwhere(board == '-')
                action = tuple(random.choice(empty_cells))

            row, col = action
            board[row, col] = current_player
            state_action_pairs.append((board_to_string(board.copy()), action))
            game_over, winner = is_game_over(board)

            if game_over:
                if winner == current_player:
                    reward = 1
                elif winner == 'draw':
                    reward = 0.5
                else:
                    reward = 0

                for state, action in state_action_pairs:
                    update_q_table(state, action, board, reward)

            else:
                current_player = players[(players.index(current_player) + 1) % num_players]

        # Gradually decrease exploration rate
        if exploration_rate > min_exploration_rate:
            exploration_rate *= exploration_decay

        # Save Q-table at regular intervals
        if (episode + 1) % save_interval == 0:
            save_q_table(q_table_file)
            print(f"Episode {episode + 1}: Q-table saved. Random actions taken: {random_count}, Q-table actions taken: {qcount}")
        

    save_q_table(q_table_file)
    print("Training complete: Q-table saved.")

# Play against the trained agent
def play_game():
    global board, current_player, game_over
    board = np.array([['-', '-', '-'],
                      ['-', '-', '-'],
                      ['-', '-', '-']])
    current_player = random.choice(players)
    game_over = False

# Function for GUI of tic-tac-toe
def on_click(row, col):
    global board, current_player, game_over, remarks_text, human_wins, ai_wins, draws

    if board[row][col] == '-' and not game_over:
        button = buttons[row][col]
        button.config(text=current_player)

        if current_player == 'X':
            button.config(fg='green')
        else:
            button.config(fg='red')

        board[row][col] = current_player
        state_action_pairs = [(board_to_string(board.copy()), (row, col))]
        game_over, winner = is_game_over(board)

        if game_over:
            if winner == 'X':
                remarks_text = "Human player wins!"
                human_wins += 1
                reward = 0
            elif winner == 'O':
                remarks_text = "Agent wins!"
                ai_wins += 1
                reward = 1
            else:
                remarks_text = "It's a draw!"
                draws += 1
                reward = 0.5

            update_remarks()
            update_win_counts()

            # Draw winning line if applicable
            winning_line = get_winning_line(board)
            if winning_line:
                draw_winning_line(winning_line)

            # Update Q-table with the final game result
            for state, action in state_action_pairs:
                update_q_table(state, action, board, reward)

        else:
            next_state = board_next_state((row, col))
            update_q_table(board_to_string(board), (row, col), next_state, 0)
            current_player = 'O' if current_player == 'X' else 'X'
            if current_player == 'O':
                action = choose_action(board, exploration_rate=0)
                row, col = action
                on_click(row, col)

# Function for giving the remarks after the game over
def update_remarks():
    remarks_label.config(text=remarks_text)

# Function to update the win counts
def update_win_counts():
    win_counts_label.config(text=f"Human Wins: {human_wins} | AI Wins: {ai_wins} | Draws: {draws}")

# Function to draw the winning line
def draw_winning_line(winning_line):
    if winning_line[0] == 'row':
        i = winning_line[1]
        for j in range(3):
            buttons[i][j].config(bg='yellow')
    elif winning_line[0] == 'col':
        i = winning_line[1]
        for j in range(3):
            buttons[j][i].config(bg='yellow')
    elif winning_line[0] == 'diag':
        if winning_line[1] == 0:
            for i in range(3):
                buttons[i][i].config(bg='yellow')
        elif winning_line[1] == 1:
            for i in range(3):
                buttons[i][2 - i].config(bg='yellow')

# Function to restart the game
def restart_game():
    global board, current_player, game_over, buttons, remarks_label, restart_button

    board = np.array([['-', '-', '-'],
                      ['-', '-', '-'],
                      ['-', '-', '-']])

    current_player = random.choice(players)
    game_over = False

    # Reset button texts and colors
    for i in range(3):
        for j in range(3):
            buttons[i][j].config(text="", fg="black", bg="SystemButtonFace")

    remarks_label.config(text="Playing...")

    # If computer starts first, make its move
    if current_player == 'O':
        action = choose_action(board, exploration_rate=0)
        row, col = action
        on_click(row, col)

# Function for start button of the tic-tac-toe
def start_game():
    global current_player, game_over, game_frame, buttons, remarks_label, restart_button, win_counts_label

    # Destroy the start button and title label
    start_button.destroy()
    title_label.destroy()

    # Create a frame for the game buttons
    game_frame = tk.Frame(root, bg='#C4A484')
    game_frame.pack(pady=20)

    buttons = [[None] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            buttons[i][j] = tk.Button(game_frame, text="", font=('Helvetica', 20), width=7, height=3,
                                      command=lambda row=i, col=j: on_click(row, col))
            buttons[i][j].grid(row=i, column=j)

    remarks_label = tk.Label(root, text="Playing ...", font=('Helvetica', 18))
    remarks_label.pack(pady=20)

    win_counts_label = tk.Label(root, text="", font=('Helvetica', 14))
    win_counts_label.pack(pady=5)

    restart_button = tk.Button(root, text="Restart Game", font=('Helvetica', 15), command=restart_game, bg='#A95C68', fg='#E6E6FA')
    restart_button.pack(pady=10)

    current_player = players[np.random.choice(num_players)]
    game_over = False

    update_win_counts()

    # If computer starts first, make its move
    if current_player == 'O':
        action = choose_action(board, exploration_rate=0)
        row, col = action
        on_click(row, col)

# Main page of the tic-tac-toe
def main_page():
    global root, start_button, title_label

    root = tk.Tk()
    root.title("Tic-Tac-Toe")
    root.geometry("700x650")
    root.configure(bg='#F2D2BD')

    center_frame = tk.Frame(root, bg='#F2D2BD')
    center_frame.pack(expand=True)

    title_label = tk.Label(center_frame, text="WELCOME TO TIC-TAC-TOE", font=('Cambria', 30, 'bold italic'), fg='#A95C68')
    title_label.pack(pady=20, side='top')

    rules_label = tk.Label(center_frame,
                           text="Rules:\n1. You are playing against a bot (computer player).\n2. The first player to get 3 of their symbols in a row (horizontally, vertically, or diagonally) wins.\n3. If all 9 squares are filled and no player has 3 in a row, the game is a draw.",
                           font=('Helvetica', 12), bg='#F2D2BD')
    rules_label.pack(pady=10)

    start_button = tk.Button(center_frame, text="Start Game", font=('Helvetica', 17), command=start_game, bg='#A95C68', fg='#E6E6FA')
    start_button.pack(side='bottom')

    root.mainloop()

if __name__ == "__main__":
    load_q_table(q_table_file)
    train_q_table()
    main_page()
    print(f"Random Count: {random_count}, Q-table Count: {qcount}.")
