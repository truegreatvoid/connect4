import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

NAME = 'Pikachu: "Pika Pika"'

# Configurações do jogo
ROWS = 6
COLS = 7
EMPTY = 0
PLAYER = 2  # Este jogador
OPPONENT = 1  # Jogador adversário

# Carregar o modelo treinado
model = load_model('connect4_model.h5')
# model = load_model('connect4agent_v3_Current_model.h5')

def jogada(board, piece):
    # Preparar o estado do tabuleiro para a rede neural
    input_board = preprocess_board(board, piece)
    
    # Fazer a previsão
    predictions = model.predict(input_board, verbose=0)[0]  # Obter as probabilidades para cada coluna
    
    # Obter as colunas ordenadas pela probabilidade
    sorted_columns = np.argsort(predictions)[::-1]
    
    # Selecionar a melhor coluna válida
    for col in sorted_columns:
        if is_valid_location(board, col):
            return col
    
    # Se nenhuma coluna for válida (empate), escolher aleatoriamente
    return np.random.randint(0, 7)

def preprocess_board(board, piece):
    """
    Preprocessa o tabuleiro para a entrada da rede neural.
    - Normaliza os valores.
    - Pode incluir mais transformações conforme necessário.
    """
    # Copiar o tabuleiro para evitar modificar o original
    board_copy = board.copy()
    
    # Opcional: Representar o tabuleiro do ponto de vista do jogador atual
    # Por exemplo, marcar as peças do oponente como -1
    board_copy[board_copy == PLAYER] = 1
    board_copy[board_copy == OPPONENT] = -1
    board_copy[board_copy == EMPTY] = 0
    
    # Achatar a matriz e normalizar
    input_board = board_copy.flatten() / 1.0  # Valores: -1, 0, 1
    
    # Converter para o formato esperado pelo modelo
    input_board = np.array([input_board])  # Batch size de 1
    
    return input_board

def is_valid_location(board, col):
    return board[ROWS - 1][col] == EMPTY
