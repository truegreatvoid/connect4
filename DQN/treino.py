import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input
from collections import deque

# Parâmetros do Connect 4
ROWS, COLS = 6, 7
EMPTY = 0
PLAYER, OPPONENT = 1, -1

# Hiperparâmetros do DQN
GAMMA = 0.95
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.99
LEARNING_RATE = 0.001
MEMORY_SIZE = 10000
BATCH_SIZE = 64

EPSILON = 0.5  # Exploração inicial

# Cria a rede neural para o DQN
def create_model():
    model = Sequential([
        Input(shape=(ROWS, COLS, 1)),
        Conv2D(32, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(COLS, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse')
    return model

# Função de pré-processamento do tabuleiro para a rede neural
def preprocess_board(board, player):
    processed_board = np.where(board == player, 1, np.where(board == -player, -1, 0))
    return processed_board.reshape(ROWS, COLS, 1)

class DQNAgent:
    def __init__(self):
        self.model = create_model()
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        valid_actions = self.get_valid_actions(state.reshape(ROWS, COLS))

        if not valid_actions:
            return 0  # Ação padrão se nenhuma válida

        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)

        q_values = self.model.predict(state.reshape(1, ROWS, COLS, 1), verbose=0)
        for action in range(COLS):
            if action not in valid_actions:
                q_values[0][action] = -np.inf  # Invalida ações não válidas
        return np.argmax(q_values[0])

    def get_valid_actions(self, board):
        return [col for col in range(COLS) if is_valid_location(board, col)]

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        
        states = np.array([sample[0] for sample in batch]).reshape(BATCH_SIZE, ROWS, COLS, 1)
        actions = np.array([sample[1] for sample in batch])
        rewards = np.array([sample[2] for sample in batch])
        next_states = np.array([sample[3] for sample in batch]).reshape(BATCH_SIZE, ROWS, COLS, 1)
        dones = np.array([sample[4] for sample in batch])

        # Calcula o target
        target = rewards + GAMMA * np.amax(self.model.predict(next_states, verbose=0), axis=1) * (~dones)
        target_f = self.model.predict(states, verbose=0)
        target_f[np.arange(BATCH_SIZE), actions] = target

        self.model.fit(states, target_f, epochs=1, verbose=0)

        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

# Funções de jogo e recompensa
def is_valid_location(board, col):
    return board[ROWS - 1, col] == EMPTY

def drop_piece(board, row, col, piece):
    board[row][col] = piece

def check_win(board, piece):
    # Horizontais
    for r in range(ROWS):
        for c in range(COLS - 3):
            if all(board[r][c + i] == piece for i in range(4)):
                return True
    # Verticais
    for r in range(ROWS - 3):
        for c in range(COLS):
            if all(board[r + i][c] == piece for i in range(4)):
                return True
    # Diagonais positivas
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if all(board[r + i][c + i] == piece for i in range(4)):
                return True
    # Diagonais negativas
    for r in range(ROWS - 3):
        for c in range(3, COLS):
            if all(board[r + i][c - i] == piece for i in range(4)):
                return True
    return False

def is_board_full(board):
    # Verifica se todas as colunas têm a célula superior ocupada
    return all(board[ROWS - 1, col] != EMPTY for col in range(COLS))

def print_board(board):
    # Inverte o tabuleiro para exibir a linha superior no topo
    print(np.flip(board, 0))

# Configuração do treinamento
agent = DQNAgent()
n_episodes = 1000  # Número de episódios de treinamento
save_interval = 50  # Intervalo de episódios para salvar o modelo

for episode in range(1, n_episodes + 1):
    board = np.zeros((ROWS, COLS), dtype=int)
    state = preprocess_board(board, PLAYER).reshape(1, ROWS, COLS, 1)
    done = False

    while not done:
        # Jogada do Agente
        action = agent.act(state)
        valid_rows = [r for r in range(ROWS) if board[r][action] == 0]
        if valid_rows:
            row = min(valid_rows)  # Alterado para min(valid_rows)
            drop_piece(board, row, action, PLAYER)
            print(f"Agente coloca na coluna {action}, linha {row}")
            print_board(board)
        else:
            print(f"A coluna {action} está cheia, escolhendo outra ação.")
            continue

        reward = 0
        if check_win(board, PLAYER):
            reward = 1
            done = True
            print("Agente venceu!")
        elif is_board_full(board):
            done = True
            print("Empate!")

        next_state = preprocess_board(board, PLAYER).reshape(1, ROWS, COLS, 1)
        agent.remember(state, action, reward, next_state, done)
        agent.train()
        state = next_state

        if done:
            break

        # Jogada do Oponente (aleatória)
        valid_opponent_actions = agent.get_valid_actions(board)
        if not valid_opponent_actions:
            print("Nenhuma ação válida disponível para o oponente. Encerrando o episódio.")
            done = True
            reward = 0  # Recompensa neutra para empate
            agent.remember(state, action, reward, next_state, done)
            agent.train()
            break

        opponent_action = random.choice(valid_opponent_actions)
        valid_rows = [r for r in range(ROWS) if board[r][opponent_action] == 0]
        if valid_rows:
            row = min(valid_rows)  # Alterado para min(valid_rows)
            drop_piece(board, row, opponent_action, OPPONENT)
            print(f"Oponente coloca na coluna {opponent_action}, linha {row}")
            print_board(board)
        else:
            print(f"A coluna {opponent_action} está cheia para o oponente, escolhendo outra ação.")
            continue

        if check_win(board, OPPONENT):
            reward = -1
            done = True
            print("Oponente venceu!")
            agent.remember(state, action, reward, next_state, done)
            agent.train()
            break
        elif is_board_full(board):
            done = True
            print("Empate!")

    # Salva o modelo a cada `save_interval` episódios
    if episode % save_interval == 0:
        agent.model.save(f"DQN/models/connect4_model_episode_{episode}.h5")
        print(f"Modelo salvo após {episode} episódios")
