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
    return None

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

def is_winning_move(board, piece, col):
    temp_board = board.copy()
    row = get_next_open_row(temp_board, col)
    if row is not None:
        drop_piece(temp_board, row, col, piece)
        return check_win(temp_board, piece)
    return False

# def choose_move(board, piece, epsilon=0.1):
#     """
#     Escolhe a próxima jogada baseada na seguinte lógica:
#     1. Se puder vencer, faça a jogada vencedora.
#     2. Se o oponente puder vencer na próxima jogada, bloqueie.
#     3. Prefira colunas centrais.
#     4. Escolha aleatoriamente entre as jogadas restantes.

#     Args:
#         board (numpy.ndarray): Estado atual do tabuleiro.
#         piece (int): Representa o jogador (1 ou 2).
#         epsilon (float): Taxa de exploração (0 a 1). Por exemplo, 0.1 significa 10% de jogadas aleatórias.

#     Returns:
#         int: Coluna escolhida para jogar.
#     """
#     valid_locations = get_valid_locations(board)
#     opponent = PLAYER1 if piece == PLAYER2 else PLAYER2

#     # 1. Verificar se pode vencer na jogada atual
#     for col in valid_locations:
#         if is_winning_move(board, piece, col):
#             return col

#     # 2. Verificar se o oponente pode vencer na próxima jogada e bloquear
#     for col in valid_locations:
#         if is_winning_move(board, opponent, col):
#             return col

#     # 3. Jogadas aleatórias com uma probabilidade epsilon
#     if random.random() < epsilon:
#         return random.choice(valid_locations)

#     # 4. Preferir colunas centrais
#     center = COLS // 2
#     preferred_columns = [center]
#     for offset in range(1, COLS//2 + 1):
#         if center - offset >= 0:
#             preferred_columns.append(center - offset)
#         if center + offset < COLS:
#             preferred_columns.append(center + offset)

#     for col in preferred_columns:
#         if col in valid_locations:
#             return col

#     # 5. Escolher aleatoriamente entre as jogadas restantes
#     return random.choice(valid_locations)

def has_three_in_a_row(board, piece):
    # Horizontal
    for c in range(COLS - 3):
        for r in range(ROWS):
            if sum([board[r][c + i] == piece for i in range(3)]) == 3 and is_valid_location(board, c + 3):
                return True

    # Vertical
    for c in range(COLS):
        for r in range(ROWS - 2):
            if sum([board[r + i][c] == piece for i in range(3)]) == 3 and get_next_open_row(board, c) is not None:
                return True

    # Diagonal positiva
    for c in range(COLS - 3):
        for r in range(ROWS - 3):
            if sum([board[r + i][c + i] == piece for i in range(3)]) == 3 and is_valid_location(board, c + 3):
                return True

    # Diagonal negativa
    for c in range(COLS - 3):
        for r in range(2, ROWS):
            if sum([board[r - i][c + i] == piece for i in range(3)]) == 3 and is_valid_location(board, c + 3):
                return True

    return False



def choose_move(board, piece, epsilon=0.1):
    """
    Escolhe a próxima jogada baseada na seguinte lógica:
    1. Se puder vencer, faça a jogada vencedora.
    2. Se o oponente puder vencer na próxima jogada, bloqueie.
    3. Priorize jogadas que levam a 3 em linha (com possibilidade de 4).
    4. Prefira colunas centrais.
    5. Escolha aleatoriamente entre as jogadas restantes.

    Args:
        board (numpy.ndarray): Estado atual do tabuleiro.
        piece (int): Representa o jogador (1 ou 2).
        epsilon (float): Taxa de exploração (0 a 1). Por exemplo, 0.1 significa 10% de jogadas aleatórias.

    Returns:
        int: Coluna escolhida para jogar.
    """
    valid_locations = get_valid_locations(board)
    opponent = PLAYER1 if piece == PLAYER2 else PLAYER2

    # 1. Verificar se pode vencer na jogada atual
    for col in valid_locations:
        if is_winning_move(board, piece, col):
            return col

    # 2. Verificar se o oponente pode vencer na próxima jogada e bloquear
    for col in valid_locations:
        if is_winning_move(board, opponent, col):
            return col

    # 3. Verificar jogadas que criariam 3 em linha, com potencial para 4 na próxima
    for col in valid_locations:
        temp_board = board.copy()
        row = get_next_open_row(temp_board, col)
        if row is not None:
            drop_piece(temp_board, row, col, piece)
            if has_three_in_a_row(temp_board, piece):
                return col

    # 4. Jogadas aleatórias com uma probabilidade epsilon
    if random.random() < epsilon:
        return random.choice(valid_locations)

    # 5. Preferir colunas centrais
    center = COLS // 2
    preferred_columns = [center]
    for offset in range(1, COLS // 2 + 1):
        if center - offset >= 0:
            preferred_columns.append(center - offset)
        if center + offset < COLS:
            preferred_columns.append(center + offset)

    for col in preferred_columns:
        if col in valid_locations:
            return col

    # 6. Escolher aleatoriamente entre as jogadas restantes
    return random.choice(valid_locations)


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

        # Escolha a jogada com base na estratégia
        col = choose_move(board, current_player, epsilon=0.1)  # 10% de jogadas aleatórias

        row = get_next_open_row(board, col)
        if row is not None:
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
        else:
            # Coluna cheia, deve ser tratada pelo mediador
            game_over = True

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
    NUM_JOGOS = 25000  # Defina o número de jogos que deseja simular
    ARQUIVO_SAIDA = 'saida/csv/connect4_data_15.csv'
    gerar_dados(NUM_JOGOS, ARQUIVO_SAIDA)
