import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, BatchNormalization, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import matplotlib.pyplot as plt

# Configurações do jogo
ROWS = 6
COLS = 7
EMPTY = 0
PLAYER1 = 1
PLAYER2 = 2

# Diretório de entrada dos dados
INPUT_DIR = 'saida/csv/gerar'

def load_all_data(input_dir):
    """
    Carrega todos os arquivos CSV do diretório especificado e os combina em um único DataFrame.

    Args:
        input_dir (str): Caminho para o diretório contendo os arquivos CSV.

    Returns:
        pandas.DataFrame: DataFrame combinado contendo todos os dados.
    """
    all_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.csv')]
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

# Aumentar os dados (opcional, caso deseje realizar data augmentation na fase de treinamento)
# Isso pode duplicar o tamanho do seu conjunto de dados, mas aumenta a diversidade.
def augment_data_during_training(X, Y):
    augmented_X = []
    augmented_Y = []
    for board, move in zip(X, Y):
        board_2d = board.reshape(ROWS, COLS)
        
        # Original
        augmented_X.append(board_2d)
        augmented_Y.append(move)
        
        # Espelhado horizontalmente
        mirrored = np.fliplr(board_2d)
        mirrored_move = COLS - 1 - move
        augmented_X.append(mirrored)
        augmented_Y.append(mirrored_move)
        
        # Rotacionar 180 graus
        rotated = np.rot90(board_2d, 2)
        rotated_move = COLS - 1 - move
        augmented_X.append(rotated)
        augmented_Y.append(rotated_move)
    
    return np.array(augmented_X), np.array(augmented_Y)

# Opcional: Aumentar os dados durante o treinamento
# X_aug, Y_aug = augment_data_during_training(X, Y)
# X = X_aug
# Y = Y_aug

# Normalizar os dados
X = X / 2.0  # Valores: 0, 0.5, 1.0 (para EMPTY, PLAYER1, PLAYER2)

# Reformatar para CNN (6x7x1)
X = X.reshape(-1, ROWS, COLS, 1)

# Dividir os dados em conjuntos de treinamento e validação
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

print(f"Treinamento - X: {X_train.shape}, Y: {Y_train.shape}")
print(f"Validação - X: {X_val.shape}, Y: {Y_val.shape}")

# Definir a arquitetura do modelo CNN com melhorias
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(ROWS, COLS, 1), padding='same', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),

    Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),

    Flatten(),
    Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(7, activation='softmax')  # 7 colunas para escolher
])

# Compilar o modelo com uma taxa de aprendizado ajustável e outros parâmetros
optimizer = Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

# Resumo do modelo
model.summary()

# Callbacks: Early Stopping, Redução da Taxa de Aprendizado e Model Checkpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
checkpoint = ModelCheckpoint('best_model_v4.h5', monitor='val_accuracy', save_best_only=True, verbose=1)

# Treinar o modelo com mais épocas e callbacks
history = model.fit(
    X_train, Y_train,
    epochs=300,  # Aumentar o número de épocas
    batch_size=128,  # Batch maior
    validation_data=(X_val, Y_val),
    callbacks=[early_stopping, reduce_lr, checkpoint],
    verbose=1
)

# Salvar o modelo treinado
model.save('connect4_model_v4_final.h5')
print("Modelo treinado e salvo como 'connect4_model_v4_final.h5'")

# Plotar o histórico de precisão e perda
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Acurácia de Treinamento')
plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
plt.title('Acurácia durante o Treinamento')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Perda de Treinamento')
plt.plot(history.history['val_loss'], label='Perda de Validação')
plt.title('Perda durante o Treinamento')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()

# Salvar e exibir os gráficos
plt.savefig('training_history_v4.png')
plt.show()
