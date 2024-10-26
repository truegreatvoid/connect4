# Exemplo com um código de treinamento fictício, carregando e re-salvando
from tensorflow.keras.models import load_model

# Carrega o modelo antigo
model = load_model('connect4agent_v3_Current_model.h5', compile=False)

# Salva o modelo no formato atualizado
model.save('connect4agent_v3_Updated_model.h5', save_format='h5')
