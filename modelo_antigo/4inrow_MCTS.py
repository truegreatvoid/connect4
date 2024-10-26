import numpy as np
import random
import math
from collections import defaultdict

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

# Verificar se o movimento atual é um movimento vencedor
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

# Função para verificar se o jogo terminou
def is_terminal_node(board):
    return winning_move(board, PLAYER_1) or winning_move(board, PLAYER_2) or len(get_valid_locations(board)) == 0

# Obter todas as localizações válidas
def get_valid_locations(board):
    valid_locations = []
    for col in range(COLS):
        if is_valid_location(board, col):
            valid_locations.append(col)
    return valid_locations

# Simular um jogo aleatório a partir do estado atual
def simulate_random_game(board, player):
    temp_board = board.copy()
    current_player = player

    while not is_terminal_node(temp_board):
        valid_locations = get_valid_locations(temp_board)
        col = random.choice(valid_locations)
        row = get_next_open_row(temp_board, col)
        drop_piece(temp_board, row, col, current_player)

        if winning_move(temp_board, current_player):
            return current_player

        current_player = PLAYER_1 if current_player == PLAYER_2 else PLAYER_2

    return None

# Implementação básica do MCTS
def mcts(board, player, simulations=1000):
    wins = defaultdict(int)
    plays = defaultdict(int)

    for _ in range(simulations):
        valid_locations = get_valid_locations(board)
        col = random.choice(valid_locations)
        temp_board = board.copy()
        row = get_next_open_row(temp_board, col)
        drop_piece(temp_board, row, col, player)

        result = simulate_random_game(temp_board, player)

        if result == player:
            wins[(player, col)] += 1
        plays[(player, col)] += 1

    # Escolha a melhor jogada com base na maior taxa de vitória
    best_col = max(valid_locations, key=lambda col: wins[(player, col)] / plays[(player, col)] if plays[(player, col)] > 0 else 0)

    return best_col

# Função principal do jogo IA vs IA com MCTS e log do vencedor
def play_game_mcts_vs_mcts(simulations=1000):
    board = create_board()
    game_over = False
    turn = 0

    print_board(board)

    while not game_over:
        if turn == 0:
            # IA Jogador 1 (MCTS)
            col = mcts(board, PLAYER_1, simulations)
            if col is not None and is_valid_location(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, PLAYER_1)

                if winning_move(board, PLAYER_1):
                    print_board(board)
                    print("Jogador 1 (IA) ganhou!")
                    game_over = True
                    break
        else:
            # IA Jogador 2 (MCTS)
            col = mcts(board, PLAYER_2, simulations)
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

# Iniciar o jogo IA vs IA com MCTS
play_game_mcts_vs_mcts(simulations=1000)
