import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np
import csv

# Carregar dados
def load_data(file_path):
    X = []
    Y = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
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

# Definir a arquitetura
model = Sequential([
    Dense(128, input_dim=42, activation='relu'),
    Dense(64, activation='relu'),
    Dense(7, activation='softmax')  # 7 colunas para escolher
])

# Compilar o modelo
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])

# Treinar o modelo
model.fit(X, Y, epochs=50, batch_size=32, validation_split=0.2)

# Salvar o modelo
model.save('connect4_model.h5')
