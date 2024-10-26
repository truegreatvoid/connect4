import numpy as np
import math

NAME = "Aluno (Minimax)"

# Configurações do jogo
ROWS = 6
COLS = 7
EMPTY = 0
PLAYER = 2  # Este jogador
OPPONENT = 1  # Jogador adversário

# Parâmetros do Minimax
WINDOW_LENGTH = 4
MAX_DEPTH = 4  # Profundidade de busca; ajuste conforme necessário

def jogada(board, piece):
    valid_locations = get_valid_locations(board)
    best_score = -math.inf
    best_col = np.random.choice(valid_locations)  # Escolhe aleatoriamente inicialmente

    for col in valid_locations:
        row = get_next_open_row(board, col)
        temp_board = board.copy()
        drop_piece(temp_board, row, col, piece)
        score = minimax(temp_board, MAX_DEPTH - 1, -math.inf, math.inf, False)
        if score > best_score:
            best_score = score
            best_col = col

    return best_col

def minimax(board, depth, alpha, beta, maximizingPlayer):
    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, PLAYER):
                return 100000
            elif winning_move(board, OPPONENT):
                return -100000
            else:  # Empate
                return 0
        else:
            return score_position(board, PLAYER)
    if maximizingPlayer:
        value = -math.inf
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, PLAYER)
            new_score = minimax(b_copy, depth - 1, alpha, beta, False)
            value = max(value, new_score)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = math.inf
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, OPPONENT)
            new_score = minimax(b_copy, depth - 1, alpha, beta, True)
            value = min(value, new_score)
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value

def get_valid_locations(board):
    valid_locations = []
    for col in range(COLS):
        if board[ROWS - 1][col] == EMPTY:
            valid_locations.append(col)
    return valid_locations

def get_next_open_row(board, col):
    for r in range(ROWS):
        if board[r][col] == EMPTY:
            return r

def drop_piece(board, row, col, piece):
    board[row][col] = piece

def is_terminal_node(board):
    return winning_move(board, PLAYER) or winning_move(board, OPPONENT) or len(get_valid_locations(board)) == 0

def winning_move(board, piece):
    # Verifica horizontal
    for c in range(COLS - 3):
        for r in range(ROWS):
            if all([board[r][c + i] == piece for i in range(WINDOW_LENGTH)]):
                return True

    # Verifica vertical
    for c in range(COLS):
        for r in range(ROWS - 3):
            if all([board[r + i][c] == piece for i in range(WINDOW_LENGTH)]):
                return True

    # Verifica diagonal positiva
    for c in range(COLS - 3):
        for r in range(ROWS - 3):
            if all([board[r + i][c + i] == piece for i in range(WINDOW_LENGTH)]):
                return True

    # Verifica diagonal negativa
    for c in range(COLS - 3):
        for r in range(3, ROWS):
            if all([board[r - i][c + i] == piece for i in range(WINDOW_LENGTH)]):
                return True

    return False

def score_position(board, piece):
    score = 0

    # Centralização
    center_array = [int(i) for i in list(board[:, COLS//2])]
    center_count = center_array.count(piece)
    score += center_count * 3

    # Pontuação para todas as janelas
    for r in range(ROWS):
        row_array = [int(i) for i in list(board[r, :])]
        for c in range(COLS - 3):
            window = row_array[c:c + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    for c in range(COLS):
        col_array = [int(i) for i in list(board[:, c])]
        for r in range(ROWS - 3):
            window = col_array[r:r + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            window = [board[r + i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            window = [board[r + 3 - i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    return score

def evaluate_window(window, piece):
    score = 0
    opp_piece = OPPONENT

    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 5
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 2

    if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        score -= 4

    return score
