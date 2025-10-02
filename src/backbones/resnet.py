# Local: src/backbones/resnet.py

from tensorflow.keras.applications import ResNet50
# --- CORREÇÃO 1: Adiciona Dropout ao import ---
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

# O import da camada CosFace permanece o mesmo
from src.losses.margin_losses import CosFace

def create_resnet50_cosface(config):
    """Cria um modelo ResNet50 com uma cabeça CosFace, configurado para fine-tuning."""
    
    # 1. Carrega o Backbone pré-treinado
    backbone = ResNet50(include_top=False, weights='imagenet', input_shape=config.INPUT_SHAPE)
    
    # --- CORREÇÃO 2: Lógica de Fine-Tuning correta ---
    # Primeiro, definimos o backbone como treinável
    backbone.trainable = True
    
    # Em seguida, congelamos as camadas iniciais.
    # O número 130 congela aproximadamente os 3 primeiros blocos da ResNet50,
    # permitindo que os blocos 4 e 5 sejam ajustados.
    for layer in backbone.layers[:-46]:
        layer.trainable = False
    
    trainable_layers = [layer.name for layer in backbone.layers if layer.trainable]
    print(f"Fine-tuning ativado. {len(trainable_layers)} camadas do backbone são treináveis.")
    # --- FIM DA CORREÇÃO ---

    # 2. Cria a cabeça do Modelo
    input_image = backbone.input
    input_label = Input(shape=(config.NUM_CLASSES,), name="label_input")
    
    x = backbone.output
    x = GlobalAveragePooling2D()(x)
    
    # Adicionamos o Dropout para regularização
    x = Dropout(0.4)(x) 
    
    x = Dense(config.EMBEDDING_SIZE, kernel_initializer='he_normal',
              kernel_regularizer=regularizers.l2(1e-4))(x)
    
    # 3. Camada CosFace
    output = CosFace(config.NUM_CLASSES, name="cosface_loss")([x, input_label])
    
    # 4. Modelo final e extrator de features
    model = Model(inputs=[input_image, input_label], outputs=output)
    feature_extractor = Model(inputs=input_image, outputs=x)
    
    return model, feature_extractor