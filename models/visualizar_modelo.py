import os
import visualkeras
from tensorflow.keras.models import load_model

# Definir caminhos
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'connect4_model_v4_final.keras')
OUTPUT_IMAGE = os.path.join(MODEL_DIR, 'visualkeras_model_v4.png')

# Carregar o modelo
model = load_model(MODEL_PATH)
print(f"Modelo carregado de: {MODEL_PATH}")

# Tentar usar scale_factor se scale não estiver disponível
try:
    visualkeras.layered_view(
        model,
        legend=True,
        scale_factor=16  # Usar scale_factor em vez de scale
    ).save(OUTPUT_IMAGE)
    print(f"Imagem do modelo gerada e salva como '{OUTPUT_IMAGE}' com scale_factor=16")
except TypeError:
    # Se scale_factor também não estiver disponível, remover o argumento
    visualkeras.layered_view(
        model,
        legend=True
    ).save(OUTPUT_IMAGE)
    print(f"Imagem do modelo gerada e salva como '{OUTPUT_IMAGE}' sem scale")
