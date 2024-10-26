import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

NAME = 'pangu'

# Configurações do jogo
ROWS = 6
COLS = 7
EMPTY = 0
PLAYER = 2  # Este jogador
OPPONENT = 1  # Jogador adversário

# Carregar o modelo treinado
model = load_model('DQN/models/connect4_model_episode_1000.h5')  # Substitua pelo caminho correto

def jogada(board, piece):
    # Preparar o estado do tabuleiro para a rede neural
    input_board = preprocess_board(board, piece)
    
    # Fazer a previsão
    q_values = model.predict(input_board, verbose=0)[0]  # Obter os valores Q para cada coluna
    
    # Definir Q-values das ações inválidas para -infinito
    valid_actions = get_valid_actions(board)
    for action in range(COLS):
        if action not in valid_actions:
            q_values[action] = -np.inf  # Invalida ações não válidas
    
    # Selecionar a melhor coluna válida
    best_action = np.argmax(q_values)
    
    # Se não houver ações válidas (empate), escolher aleatoriamente
    if best_action == -1 or q_values[best_action] == -np.inf:
        return np.random.choice(valid_actions) if valid_actions else np.random.randint(0, COLS)
    
    return best_action

def preprocess_board(board, piece):
    """
    Preprocessa o tabuleiro para a entrada da rede neural.
    - Representa o tabuleiro do ponto de vista do jogador atual.
    """
    processed_board = np.where(board == piece, 1, np.where(board == OPPONENT, -1, 0))
    return processed_board.reshape(1, ROWS, COLS, 1)

def get_valid_actions(board):
    """Retorna as colunas válidas para um movimento."""
    return [col for col in range(COLS) if is_valid_location(board, col)]

def is_valid_location(board, col):
    """Verifica se uma coluna específica está disponível para uma jogada."""
    return board[ROWS - 1][col] == EMPTY

def get_next_open_row(board, col):
    """Retorna a próxima linha disponível na coluna escolhida."""
    for r in range(ROWS):
        if board[r][col] == EMPTY:
            return r
    return None

def drop_piece(board, row, col, piece):
    """Coloca a peça na posição especificada."""
    board[row][col] = piece

def check_win(board, piece):
    # Verifica vitórias horizontais
    for c in range(COLS - 3):
        for r in range(ROWS):
            if all(board[r][c + i] == piece for i in range(4)):
                return True

    # Verifica vitórias verticais
    for c in range(COLS):
        for r in range(ROWS - 3):
            if all(board[r + i][c] == piece for i in range(4)):
                return True

    # Verifica vitórias diagonais positivas
    for c in range(COLS - 3):
        for r in range(ROWS - 3):
            if all(board[r + i][c + i] == piece for i in range(4)):
                return True

    # Verifica vitórias diagonais negativas
    for c in range(COLS - 3):
        for r in range(3, ROWS):
            if all(board[r - i][c + i] == piece for i in range(4)):
                return True

    return False

def is_board_full(board):
    """Verifica se o tabuleiro está cheio."""
    return all(board[ROWS - 1][col] != EMPTY for col in range(COLS))

def print_board(board):
    """Imprime o tabuleiro no console."""
    print(np.flip(board, 0))

if __name__ == "__main__":
    board = np.zeros((ROWS, COLS), dtype=int)
    game_over = False
    turn = 0  # 0 para o agente, 1 para o oponente

    while not game_over:
        if turn == 0:
            # Jogada do Agente
            col = jogada(board, PLAYER)
            row = get_next_open_row(board, col)
            if row is not None:
                drop_piece(board, row, col, PLAYER)
                print(f"Agente coloca na coluna {col}")
                print_board(board)

                if check_win(board, PLAYER):
                    print("Agente venceu!")
                    game_over = True
                elif is_board_full(board):
                    print("Empate!")
                    game_over = True

                turn = 1
            else:
                print(f"Agente tentou jogar na coluna {col}, mas está cheia.")
                game_over = True

        else:
            # Jogada do Oponente (aleatória)
            valid_actions = get_valid_actions(board)
            if valid_actions:
                col = np.random.choice(valid_actions)
                row = get_next_open_row(board, col)
                if row is not None:
                    drop_piece(board, row, col, OPPONENT)
                    print(f"Oponente coloca na coluna {col}")
                    print_board(board)

                    if check_win(board, OPPONENT):
                        print("Oponente venceu!")
                        game_over = True
                    elif is_board_full(board):
                        print("Empate!")
                        game_over = True

                    turn = 0
                else:
                    print(f"Oponente tentou jogar na coluna {col}, mas está cheia.")
                    game_over = True
            else:
                print("Empate!")
                game_over = True
