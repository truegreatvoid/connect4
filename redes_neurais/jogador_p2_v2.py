import numpy as np
from tensorflow.keras.models import load_model

NAME = 'tiago'


# Carregar o modelo
model = load_model('models/connect4_model_v4_final.keras')

def preprocess_board(board):
    """
    Preprocessa o tabuleiro para corresponder à forma esperada pelo modelo.
    """
    board = np.array(board)
    board = board / 2.0
    board = board.reshape(6, 7)
    board = np.expand_dims(board, axis=-1)  # (6,7,1)
    board = np.expand_dims(board, axis=0)   # (1,6,7,1)
    return board

def jogada(board, current_player):
    """
    Determina a próxima jogada do jogador 2 usando o modelo treinado.
    
    Args:
        board (list ou np.ndarray): Representação do tabuleiro como uma lista ou array achatado de 42 elementos.
        current_player (int): Identificador do jogador atual.
        
    Returns:
        int: Coluna escolhida para jogar.
    """
    # Preprocessar o tabuleiro
    input_board = preprocess_board(board)
    
    # Fazer a previsão
    predictions = model.predict(input_board, verbose=0)[0]  # (7,)
    
    # Selecionar a coluna com a maior probabilidade
    coluna_escolhida = np.argmax(predictions)
    
    return coluna_escolhida
