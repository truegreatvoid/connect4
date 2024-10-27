import numpy as np
import pandas as pd
import os
import csv
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Configurações do jogo
ROWS = 6
COLS = 7
EMPTY = 0
PLAYER1 = 1
PLAYER2 = 2

# Diretório de entrada dos dados
INPUT_DIR = 'saida/dados'

def load_all_data(input_dir):
    """
    Carrega todos os arquivos CSV do diretório especificado e os combina em um único DataFrame.

    Args:
        input_dir (str): Caminho para o diretório contendo os arquivos CSV.

    Returns:
        pandas.DataFrame: DataFrame combinado contendo todos os dados.
    """
    all_files = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir)
        if f.endswith('.csv')
    ]
    df_list = []
    for file in all_files:
        print(f'Carregando arquivo: {file}')
        df = pd.read_csv(file)
        df_list.append(df)
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

# Carregar todos os dados
data = load_all_data(INPUT_DIR)

# Separar features e labels
X = data.iloc[:, :-1].values  # Primeiras 42 colunas
Y = data.iloc[:, -1].values   # Última coluna 'move'

# # Função para carregar os dados
# def load_data(file_path):
#     X = []
#     Y = []
#     with open(file_path, 'r') as file:
#         reader = csv.reader(file)
#         header = next(reader)  # Pular o cabeçalho
#         for row in reader:
#             board = list(map(int, row[:-1]))
#             move = int(row[-1])
#             X.append(board)
#             Y.append(move)
#     return np.array(X), np.array(Y)

# Carregar os dados
# X, Y = load_data('saida\csv\connect4_data.csv')

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

# Definir a arquitetura do modelo
model = Sequential([
    Dense(128, input_dim=42, activation='relu'),
    Dense(64, activation='relu'),
    Dense(7, activation='softmax')  # 7 colunas para escolher
])

# Compilar o modelo
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# Resumo do modelo
model.summary()

# Treinar o modelo
history = model.fit(
    X_train, Y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, Y_val)
)

# Salvar o modelo treinado
model.save('connect4_model_v1_v2_m1.h5')
print("Modelo treinado e salvo como 'connect4_model_v1_v2_m1.h5'")
