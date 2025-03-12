import numpy as np
from flask import Flask, request, jsonify, render_template
import random
import pickle
import os

app = Flask(__name__)

# Defining global variables
players = ['X', 'O']
num_players = len(players)
Q = {}
learning_rate = 0.2
discount_factor = 0.9
exploration_rate = 1.0
exploration_decay = 0.999
min_exploration_rate = 0.01
num_episodes = 200000
save_interval = 1000
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/move', methods=['POST'])
def move():
    global board, current_player, game_over

    data = request.json
    board = np.array(data['board'])
    current_player = data['player']

    if current_player == 'X':
        row, col = data['move']
        board[row, col] = current_player
        game_over, winner = is_game_over(board)
        if game_over:
            return jsonify({'board': board.tolist(), 'winner': winner})
        else:
            current_player = 'O'
            # AI makes a move
            action = choose_action(board, exploration_rate=0)
            row, col = action
            board[row, col] = current_player
            game_over, winner = is_game_over(board)
            return jsonify({'board': board.tolist(), 'winner': winner})

    return jsonify({'board': board.tolist(), 'winner': None})

@app.route('/restart', methods=['POST'])
def restart():
    global board, current_player, game_over

    board = np.array([['-', '-', '-'],
                      ['-', '-', '-'],
                      ['-', '-', '-']])
    current_player = random.choice(players)
    game_over = False

    if current_player == 'O':
        action = choose_action(board, exploration_rate=0)
        row, col = action
        board[row, col] = current_player

    return jsonify({'board': board.tolist(), 'player': current_player})

if __name__ == "__main__":
    load_q_table(q_table_file)
    #train_q_table()
    app.run(debug=True)
