import numpy as np

# Definições do jogo
ROWS = 6
COLS = 7
PLAYER_1 = 1
PLAYER_2 = 2

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

# Verificar se há 4 peças consecutivas para ganhar (horizontal, vertical, diagonal)
def winning_move(board, piece):
    # Verificar horizontais
    for c in range(COLS-3):
        for r in range(ROWS):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                return True

    # Verificar verticais
    for c in range(COLS):
        for r in range(ROWS-3):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                return True

    # Verificar diagonais ascendentes
    for c in range(COLS-3):
        for r in range(ROWS-3):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                return True

    # Verificar diagonais descendentes
    for c in range(COLS-3):
        for r in range(3, ROWS):
            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                return True

# Função principal do jogo
def play_game():
    board = create_board()
    game_over = False
    turn = 0

    print_board(board)

    while not game_over:
        # Alternar entre jogador 1 e jogador 2
        if turn == 0:
            col = int(input("Jogador 1, faça sua jogada (0-6): "))
            if is_valid_location(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, PLAYER_1)

                if winning_move(board, PLAYER_1):
                    print_board(board)
                    print("Jogador 1 ganhou!")
                    game_over = True
        else:
            col = int(input("Jogador 2, faça sua jogada (0-6): "))
            if is_valid_location(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, PLAYER_2)

                if winning_move(board, PLAYER_2):
                    print_board(board)
                    print("Jogador 2 ganhou!")
                    game_over = True

        print_board(board)

        turn += 1
        turn = turn % 2

# Iniciar o jogo
play_game()
