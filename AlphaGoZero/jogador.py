import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import math

# Nome do jogador e configurações do jogo
NAME = 'Pikachu: "Pika Pika"'
ROWS, COLS, EMPTY = 6, 7, 0
PLAYER, OPPONENT = 2, 1  # Definição de peças

# Carrega o modelo treinado
model = load_model('connect4agent_v3_Current_model.h5')
model.summary()
# Parâmetros do MCTS
N_SIMULATIONS = 50  # Número de simulações de MCTS para cada jogada

def preprocess_board(board, piece):
    """Prepara o tabuleiro para entrada no modelo de rede neural."""
    board_copy = board.copy()
    board_copy[board_copy == PLAYER] = 1
    board_copy[board_copy == OPPONENT] = -1
    board_copy[board_copy == EMPTY] = 0
    # input_board = board_copy.flatten() / 1.0  # Normaliza o tabuleiro
    input_board = board_copy.flatten().reshape((1, 42))  # para uma entrada plana de (42,)

    return np.array([input_board])  # Batch de tamanho 1

def is_valid_location(board, col):
    """Verifica se a coluna escolhida é válida."""
    return board[ROWS - 1][col] == EMPTY

def get_possible_moves(board):
    """Retorna as colunas válidas para um movimento."""
    return [col for col in range(COLS) if is_valid_location(board, col)]

def mcts_predict(board, piece):
    """Realiza a busca com MCTS e retorna a melhor jogada."""
    # Inicializa contagens e valores
    counts = np.zeros(COLS)  # Conta visitas a cada movimento
    values = np.zeros(COLS)  # Acumula valores para cada movimento

    for _ in range(N_SIMULATIONS):
        # Expande cada possível movimento
        col = monte_carlo_tree_search(board, piece, counts, values)
        counts[col] += 1  # Incrementa o número de visitas à coluna escolhida

    # Seleciona o movimento mais visitado
    best_col = np.argmax(counts)
    return best_col

def monte_carlo_tree_search(board, piece, counts, values):
    """Realiza uma simulação de MCTS para encontrar a melhor jogada."""
    possible_moves = get_possible_moves(board)
    best_move = None
    best_ucb = -math.inf

    for col in possible_moves:
        # Calcula Upper Confidence Bound (UCB) para cada movimento
        ucb = values[col] / (counts[col] + 1) + math.sqrt(math.log(sum(counts) + 1) / (counts[col] + 1))
        
        if ucb > best_ucb:
            best_ucb = ucb
            best_move = col

    # Simula o movimento selecionado
    next_board = board.copy()
    row = np.max(np.where(next_board[:, best_move] == EMPTY))
    next_board[row, best_move] = piece

    # Predição da rede neural para política e valor
    input_board = preprocess_board(next_board, piece)
    policy, value = model.predict(input_board, verbose=0)

    # Atualiza valor de acordo com a previsão da rede neural
    values[best_move] += value[0] if piece == PLAYER else -value[0]
    return best_move

def jogada(board, piece):
    """Escolhe uma jogada utilizando a MCTS."""
    return mcts_predict(board, piece)
