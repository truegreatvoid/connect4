import numpy as np
import csv
import random

# Configurações do tabuleiro
ROWS = 6
COLS = 7
EMPTY = 0
PLAYER1 = 1
PLAYER2 = 2

def create_board():
    return np.zeros((ROWS, COLS), dtype=int)

def is_valid_location(board, col):
    return board[ROWS - 1][col] == EMPTY

def get_valid_locations(board):
    valid_locations = []
    for col in range(COLS):
        if is_valid_location(board, col):
            valid_locations.append(col)
    return valid_locations

def get_next_open_row(board, col):
    for r in range(ROWS):
        if board[r][col] == EMPTY:
            return r

def drop_piece(board, row, col, piece):
    board[row][col] = piece

def check_win(board, piece):
    # Verifica horizontal, vertical e diagonais
    # Horizontal
    for c in range(COLS - 3):
        for r in range(ROWS):
            if all([board[r][c + i] == piece for i in range(4)]):
                return True

    # Vertical
    for c in range(COLS):
        for r in range(ROWS - 3):
            if all([board[r + i][c] == piece for i in range(4)]):
                return True

    # Diagonal positiva
    for c in range(COLS - 3):
        for r in range(ROWS - 3):
            if all([board[r + i][c + i] == piece for i in range(4)]):
                return True

    # Diagonal negativa
    for c in range(COLS - 3):
        for r in range(3, ROWS):
            if all([board[r - i][c + i] == piece for i in range(4)]):
                return True

    return False

def simulate_game():
    board = create_board()
    game_over = False
    turn = random.randint(0, 1)  # 0 para PLAYER1, 1 para PLAYER2
    game_data = []

    while not game_over:
        current_player = PLAYER1 if turn == 0 else PLAYER2
        valid_locations = get_valid_locations(board)
        
        if not valid_locations:
            # Empate
            break

        # Escolha aleatória da coluna
        col = random.choice(valid_locations)
        row = get_next_open_row(board, col)

        # Registrar o estado antes da jogada
        board_state = board.copy()
        # Flatten o tabuleiro para 42 elementos
        board_flat = board_state.flatten().tolist()
        game_data.append(board_flat + [col])

        # Executar a jogada
        drop_piece(board, row, col, current_player)

        # Verificar vitória
        if check_win(board, current_player):
            game_over = True

        # Alternar turno
        turn = 1 - turn

    return game_data

def gerar_dados(num_jogos, arquivo_saida):
    with open(arquivo_saida, 'w', newline='') as file:
        writer = csv.writer(file)
        # Escrever o cabeçalho (opcional)
        header = [f'c{i}' for i in range(42)] + ['move']
        writer.writerow(header)

        for jogo in range(num_jogos):
            jogo_data = simulate_game()
            writer.writerows(jogo_data)
            if (jogo + 1) % 1000 == 0:
                print(f'{jogo + 1} jogos simulados.')

    print(f'Dados gerados e salvos em {arquivo_saida}.')

if __name__ == "__main__":
    NUM_JOGOS = 10000  # Defina o número de jogos que deseja simular
    ARQUIVO_SAIDA = 'connect4_data.csv'
    gerar_dados(NUM_JOGOS, ARQUIVO_SAIDA)
