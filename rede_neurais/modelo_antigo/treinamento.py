from tensorflow.keras.layers import Conv2D, MaxPooling2D

# Redefinir X para ter a forma (amostras, linhas, colunas, canais)
X = X.reshape(-1, 6, 7, 1)  # 1 canal para o tabuleiro

model = Sequential([
    Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(6, 7, 1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(7, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])

model.fit(X, Y, epochs=50, batch_size=32, validation_split=0.2)

model.save('connect4_cnn_model.h5')
