import numpy as np
import csv
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt

# Configurações do jogo
ROWS = 6
COLS = 7
EMPTY = 0
PLAYER1 = 1
PLAYER2 = 2

# Função para carregar os dados
def load_data(file_path):
    X = []
    Y = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Pular o cabeçalho
        for row in reader:
            board = list(map(int, row[:-1]))
            move = int(row[-1])
            X.append(board)
            Y.append(move)
    return np.array(X), np.array(Y)

# Carregar os dados
X, Y = load_data('saida/csv/connect4_data_v4.csv')

# Verificar a forma dos dados
print(f"Formato de X: {X.shape}")
print(f"Formato de Y: {Y.shape}")

# Normalizar os dados
X = X / 2.0  # Valores: 0, 0.5, 1.0 (para EMPTY, PLAYER1, PLAYER2)

# Dividir os dados em conjuntos de treinamento e validação
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Verificar a divisão dos dados
print(f"Treinamento - X: {X_train.shape}, Y: {Y_train.shape}")
print(f"Validação - X: {X_val.shape}, Y: {Y_val.shape}")

# Definir a arquitetura do modelo com melhorias
model = Sequential([
    Dense(256, input_dim=42, activation='relu'),
    Dropout(0.3),  # Dropout para evitar sobreajuste
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(7, activation='softmax')  # 7 colunas para escolher
])

# Compilar o modelo com uma taxa de aprendizado menor
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(learning_rate=0.0005),  # Taxa de aprendizado ajustada
    metrics=['accuracy']
)

# Resumo do modelo
model.summary()

# Callback para parada antecipada
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Treinar o modelo com mais épocas e callbacks
history = model.fit(
    X_train, Y_train,
    epochs=150,  # Aumentar o número de épocas
    batch_size=64,  # Batch maior
    validation_data=(X_val, Y_val),
    callbacks=[early_stopping]
)

# Salvar o modelo treinado
model.save('connect4_model_v2.h5')
print("Modelo treinado e salvo como 'connect4_model_v2.h5'")


# Plotar o histórico de precisão
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Acurácia de Treinamento')
plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
plt.title('Acurácia durante o Treinamento')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()

# Plotar o histórico de perda
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Perda de Treinamento')
plt.plot(history.history['val_loss'], label='Perda de Validação')
plt.title('Perda durante o Treinamento')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()

# Salvar e exibir os gráficos
plt.savefig('training_history_v2.png')
plt.show()