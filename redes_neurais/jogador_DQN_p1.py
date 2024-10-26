import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from tensorflow.keras.losses import MeanSquaredError

NAME = 'Zhurong'

# Configurações do jogo
ROWS = 6
COLS = 7
EMPTY = 0
PLAYER = 2      # Este jogador (DQN)
OPPONENT = 1    # Jogador adversário

# Caminho para o diretório dos modelos
MODEL_DIR = 'DQN/models'  # Ajuste conforme necessário

def get_latest_model_path(model_dir):
    """
    Retorna o caminho para o modelo mais recente baseado no número de episódios.

    Args:
        model_dir (str): Diretório onde os modelos estão salvos.

    Returns:
        str: Caminho completo para o modelo mais recente.
    """
    models = [f for f in os.listdir(model_dir) if f.endswith('.h5') or f.endswith('.keras')]
    if not models:
        raise FileNotFoundError("Nenhum modelo encontrado no diretório especificado.")
    
    # Ordenar os modelos pelo número de episódios extraído do nome do arquivo
    def extract_episode_number(filename):
        try:
            return int(filename.split('_')[-1].split('.')[0])
        except ValueError:
            return -1  # Retorna -1 para arquivos que não seguem o padrão

    models.sort(key=extract_episode_number)
    latest_model = models[-1]
    return os.path.join(model_dir, latest_model)

# Carregar o modelo treinado
model_path = get_latest_model_path(MODEL_DIR)
# model = load_model(model_path)
model = load_model(model_path, custom_objects={'mse': MeanSquaredError()})

print(f"Modelo carregado: {model_path}")

def jogada(board, piece):
    """
    Decide a melhor jogada para o agente DQN baseado no estado atual do tabuleiro.

    Args:
        board (numpy.ndarray): Estado atual do tabuleiro de Connect 4.
        piece (int): Representa o jogador (1 ou 2).

    Returns:
        int: Coluna escolhida para jogar.
    """
    # Pré-processar o tabuleiro
    input_board = preprocess_board(board, piece)
    
    # Fazer a previsão dos valores Q para cada coluna
    q_values = model.predict(input_board, verbose=0)[0]  # Shape: (7,)
    
    # Obter as colunas válidas
    valid_actions = get_valid_actions(board)
    
    if not valid_actions:
        # Se não houver ações válidas, retornar uma coluna aleatória (embora isso deva ser tratado pelo mediador)
        print("Nenhuma ação válida disponível para o agente.")
        return np.random.randint(0, COLS)
    
    # Definir Q-values das ações inválidas para -infinito para evitar a escolha
    for action in range(COLS):
        if action not in valid_actions:
            q_values[action] = -np.inf
    
    # Selecionar a coluna com o maior valor Q entre as válidas
    best_action = np.argmax(q_values)
    
    # Caso todas as ações válidas tenham Q-value -inf (não deveria acontecer), escolher aleatoriamente
    if q_values[best_action] == -np.inf:
        best_action = np.random.choice(valid_actions)
    
    print(f"{NAME} escolheu a coluna: {best_action}")
    return best_action

def preprocess_board(board, piece):
    """
    Pré-processa o tabuleiro para a entrada da rede neural.

    Args:
        board (numpy.ndarray): Estado atual do tabuleiro.
        piece (int): Representa o jogador (1 ou 2).

    Returns:
        numpy.ndarray: Tabuleiro pré-processado no formato (1, 6, 7, 1).
    """
    # Mapear as peças do jogador atual para 1 e do oponente para -1
    processed_board = np.where(board == piece, 1, np.where(board == get_opponent(piece), -1, 0))
    # Reshape para (1, 6, 7, 1)
    return processed_board.reshape(1, ROWS, COLS, 1)

def get_valid_actions(board):
    """
    Retorna as colunas válidas para um movimento.

    Args:
        board (numpy.ndarray): Estado atual do tabuleiro.

    Returns:
        list: Lista de colunas válidas (0 a 6).
    """
    return [col for col in range(COLS) if is_valid_location(board, col)]

def is_valid_location(board, col):
    """
    Verifica se uma coluna específica está disponível para uma jogada.

    Args:
        board (numpy.ndarray): Estado atual do tabuleiro.
        col (int): Coluna a ser verificada.

    Returns:
        bool: True se a coluna estiver disponível, False caso contrário.
    """
    return board[ROWS - 1][col] == EMPTY

def get_opponent(piece):
    """
    Retorna a peça do oponente baseada na peça atual.

    Args:
        piece (int): Peça atual (1 ou 2).

    Returns:
        int: Peça do oponente.
    """
    return PLAYER if piece == OPPONENT else OPPONENT

# Função opcional para exibir o tabuleiro (útil para depuração)
def print_board(board):
    print(np.flip(board, 0))

# Funções auxiliares para teste independente (opcional)
if __name__ == "__main__":
    # Exemplo de tabuleiro vazio
    test_board = np.zeros((ROWS, COLS), dtype=int)
    chosen_col = jogada(test_board, PLAYER)
    print(f"Coluna escolhida pelo agente: {chosen_col}")
