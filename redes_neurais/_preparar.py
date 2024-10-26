import numpy as np
import csv
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_data(file_path):
    X = []
    Y = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Pular o cabeçalho
        for row in reader:
            board = list(map(int, row[:-1]))
            move = int(row[-1])
            X.append(board)
            Y.append(move)
    return np.array(X), np.array(Y)

# Carregar os dados
X, Y = load_data('connect4_data.csv')

# Normalizar os dados
X = X / 2.0  # Valores: 0, 0.5, 1.0 (para EMPTY, PLAYER1, PLAYER2)

# Dividir em conjuntos de treinamento e validação
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
