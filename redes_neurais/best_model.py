from tensorflow.keras.models import load_model

# Carregar o melhor modelo salvo
best_model = load_model(os.path.join(MODEL_DIR, 'best_model_v4.h5'))

# Avaliar o modelo no conjunto de validação
loss, accuracy = best_model.evaluate(X_val, Y_val, verbose=0)
print(f'Best Model - Loss: {loss}, Accuracy: {accuracy}')
