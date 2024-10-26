import numpy as np
import random
import math

# Definições do jogo
ROWS = 6
COLS = 7
PLAYER_1 = 1
PLAYER_2 = 2
EMPTY = 0
WINDOW_LENGTH = 4

# Criar o tabuleiro
def create_board():
    return np.zeros((ROWS, COLS))

# Soltar peça
def drop_piece(board, row, col, piece):
    board[row][col] = piece

# Verificar se a coluna tem espaço
def is_valid_location(board, col):
    return board[ROWS-1][col] == 0

# Encontrar a próxima linha disponível na coluna
def get_next_open_row(board, col):
    for r in range(ROWS):
        if board[r][col] == 0:
            return r

# Imprimir o tabuleiro
def print_board(board):
    print(np.flip(board, 0))

def winning_move(board, piece):
    # Verificar horizontais
    for c in range(COLS - 3):
        for r in range(ROWS):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                print(f"Sequência horizontal de {piece} encontrada na linha {r} começando na coluna {c}.")
                return True

    # Verificar verticais
    for c in range(COLS):
        for r in range(ROWS - 3):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                print(f"Sequência vertical de {piece} encontrada na coluna {c} começando na linha {r}.")
                return True

    # Verificar diagonais ascendentes (da esquerda para a direita)
    for c in range(COLS - 3):
        for r in range(ROWS - 3):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                print(f"Sequência diagonal ascendente de {piece} encontrada começando na linha {r}, coluna {c}.")
                return True

    # Verificar diagonais descendentes (da esquerda para a direita)
    for c in range(COLS - 3):
        for r in range(3, ROWS):
            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                print(f"Sequência diagonal descendente de {piece} encontrada começando na linha {r}, coluna {c}.")
                return True

    return False


# Verifica se o jogo terminou (empate)
def is_terminal_node(board):
    return winning_move(board, PLAYER_1) or winning_move(board, PLAYER_2) or len(get_valid_locations(board)) == 0

# Avaliar o estado do tabuleiro para um jogador
def evaluate_window(window, piece):
    score = 0
    opp_piece = PLAYER_1 if piece == PLAYER_2 else PLAYER_2

    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 5
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 2

    if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        score -= 4

    return score

# Avaliar o tabuleiro
def score_position(board, piece):
    score = 0

    # Pontuar o centro do tabuleiro
    center_array = [int(i) for i in list(board[:, COLS//2])]
    center_count = center_array.count(piece)
    score += center_count * 3

    # Pontuar horizontais
    for r in range(ROWS):
        row_array = [int(i) for i in list(board[r,:])]
        for c in range(COLS-3):
            window = row_array[c:c+WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    # Pontuar verticais
    for c in range(COLS):
        col_array = [int(i) for i in list(board[:,c])]
        for r in range(ROWS-3):
            window = col_array[r:r+WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    # Pontuar diagonais ascendentes
    for r in range(ROWS-3):
        for c in range(COLS-3):
            window = [board[r+i][c+i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    # Pontuar diagonais descendentes
    for r in range(ROWS-3):
        for c in range(COLS-3):
            window = [board[r+3-i][c+i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    return score

# Obter todas as localizações válidas
def get_valid_locations(board):
    valid_locations = []
    for col in range(COLS):
        if is_valid_location(board, col):
            valid_locations.append(col)
    return valid_locations

# Escolher a melhor jogada para a IA
def pick_best_move(board, piece):
    valid_locations = get_valid_locations(board)
    best_score = -10000
    best_col = random.choice(valid_locations)
    for col in valid_locations:
        row = get_next_open_row(board, col)
        temp_board = board.copy()
        drop_piece(temp_board, row, col, piece)
        score = score_position(temp_board, piece)
        if score > best_score:
            best_score = score
            best_col = col
    return best_col

# # Implementar o algoritmo Minimax com poda Alpha-Beta
# def minimax(board, depth, alpha, beta, maximizingPlayer):
#     valid_locations = get_valid_locations(board)
#     is_terminal = is_terminal_node(board)
#     if depth == 0 or is_terminal:
#         if is_terminal:
#             if winning_move(board, PLAYER_2):
#                 return (None, 100000000000000)
#             elif winning_move(board, PLAYER_1):
#                 return (None, -10000000000000)
#             else:  # Empate
#                 return (None, 0)
#         else:  # Profundidade 0
#             return (None, score_position(board, PLAYER_2))
#     if maximizingPlayer:
#         value = -math.inf
#         best_col = random.choice(valid_locations)
#         for col in valid_locations:
#             row = get_next_open_row(board, col)
#             temp_board = board.copy()
#             drop_piece(temp_board, row, col, PLAYER_2)
#             new_score = minimax(temp_board, depth-1, alpha, beta, False)[1]
#             if new_score > value:
#                 value = new_score
#                 best_col = col
#             alpha = max(alpha, value)
#             if alpha >= beta:
#                 break
#         return best_col, value
#     else:  # Minimizing player
#         value = math.inf
#         best_col = random.choice(valid_locations)
#         for col in valid_locations:
#             row = get_next_open_row(board, col)
#             temp_board = board.copy()
#             drop_piece(temp_board, row, col, PLAYER_1)
#             new_score = minimax(temp_board, depth-1, alpha, beta, True)[1]
#             if new_score < value:
#                 value = new_score
#                 best_col = col
#             beta = min(beta, value)
#             if alpha >= beta:
#                 break
#         return best_col, value




def minimax(board, depth, alpha, beta, maximizingPlayer):
    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)
    
    # Verifica se o jogo acabou (vencedor ou empate)
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, PLAYER_2):
                return (None, 100000000000000)
            elif winning_move(board, PLAYER_1):
                return (None, -10000000000000)
            else:  # Empate
                return (None, 0)
        else:  # Profundidade 0
            return (None, score_position(board, PLAYER_2))

    if maximizingPlayer:
        value = -math.inf
        best_cols = []
        for col in valid_locations:
            row = get_next_open_row(board, col)
            temp_board = board.copy()
            drop_piece(temp_board, row, col, PLAYER_2)
            new_score = minimax(temp_board, depth-1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                best_cols = [col]
            elif new_score == value:
                best_cols.append(col)

            alpha = max(alpha, value)
            if alpha >= beta:
                break
        best_col = random.choice(best_cols)  # Escolha aleatória entre as melhores colunas
        return best_col, value
    else:
        value = math.inf
        best_cols = []
        for col in valid_locations:
            row = get_next_open_row(board, col)
            temp_board = board.copy()
            drop_piece(temp_board, row, col, PLAYER_1)
            new_score = minimax(temp_board, depth-1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                best_cols = [col]
            elif new_score == value:
                best_cols.append(col)

            beta = min(beta, value)
            if alpha >= beta:
                break
        best_col = random.choice(best_cols)
        return best_col, value


# Verifica se o jogo acabou (empate ou vitória)
def is_terminal_node(board):
    return winning_move(board, PLAYER_1) or winning_move(board, PLAYER_2) or len(get_valid_locations(board)) == 0

# Obtém todas as colunas válidas para jogar
def get_valid_locations(board):
    valid_locations = []
    for col in range(COLS):
        if is_valid_location(board, col):
            valid_locations.append(col)
    return valid_locations

# # Verifica se a jogada é válida antes de fazer a jogada
# def play_game_ai_vs_ai():
#     board = create_board()
#     game_over = False
#     turn = 0

#     print_board(board)

#     while not game_over:
#         if turn == 0:
#             # IA Jogador 1 (Minimizing Player)
#             col, minimax_score = minimax(board, 4, -math.inf, math.inf, False)
#             if col is not None and is_valid_location(board, col):
#                 row = get_next_open_row(board, col)
#                 drop_piece(board, row, col, PLAYER_1)

#                 if winning_move(board, PLAYER_1):
#                     print_board(board)
#                     print("Jogador 1 (IA) ganhou!")
#                     game_over = True
#         else:
#             # IA Jogador 2 (Maximizing Player)
#             col, minimax_score = minimax(board, 4, -math.inf, math.inf, True)
#             if col is not None and is_valid_location(board, col):
#                 row = get_next_open_row(board, col)
#                 drop_piece(board, row, col, PLAYER_2)

#                 if winning_move(board, PLAYER_2):
#                     print_board(board)
#                     print("Jogador 2 (IA) ganhou!")
#                     game_over = True

#         if len(get_valid_locations(board)) == 0 and not game_over:
#             print("Empate!")
#             game_over = True

#         print_board(board)

#         turn += 1
#         turn = turn % 2

# Função principal do jogo IA vs IA
def play_game_ai_vs_ai():
    board = create_board()
    game_over = False
    turn = 0

    print_board(board)

    while not game_over:
        if turn == 0:
            # IA Jogador 1 (Minimizing Player)
            col, minimax_score = minimax(board, 4, -math.inf, math.inf, False)
            if col is not None and is_valid_location(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, PLAYER_1)

                if winning_move(board, PLAYER_1):
                    print_board(board)
                    print("Jogador 1 (IA) ganhou!")
                    game_over = True
                    break
        else:
            # IA Jogador 2 (Maximizing Player)
            col, minimax_score = minimax(board, 4, -math.inf, math.inf, True)
            if col is not None and is_valid_location(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, PLAYER_2)

                if winning_move(board, PLAYER_2):
                    print_board(board)
                    print("Jogador 2 (IA) ganhou!")
                    game_over = True
                    break

        if len(get_valid_locations(board)) == 0 and not game_over:
            print("Empate!")
            game_over = True
            break

        print_board(board)

        turn += 1
        turn = turn % 2

# Iniciar o jogo IA vs IA
play_game_ai_vs_ai()

