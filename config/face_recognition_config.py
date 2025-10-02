# config/face_recognition_config.py
import os


DATA_ROOT = '/home/ubuntu/noleak/face_embeddings/src/data'
VGGFACE2_TRAIN_PATH = os.path.join(DATA_ROOT, '/home/ubuntu/noleak/face_embeddings/data/raw/vggface2_112x112')
LFW_PATH = os.path.join(DATA_ROOT, '/home/ubuntu/noleak/face_embeddings/data/raw/lfw')
LFW_PAIRS_PATH = os.path.join(DATA_ROOT, '/home/ubuntu/noleak/face_embeddings/data/raw/lfw/lfw_ann.txt')

# --- Parâmetros do Modelo e Treinamento ---
class FaceRecognitionConfig:
    NUM_CLASSES = len(os.listdir(VGGFACE2_TRAIN_PATH))

    # Dimensões da Imagem e Embedding
    IMAGE_SIZE = (112, 112)
    INPUT_SHAPE = (112, 112, 3)
    EMBEDDING_SIZE = 512 # Tamanho padrão para embeddings de face

    # Define a fração do dataset a ser usada (ex: 0.1 = 10%). 
    # Mude para 1.0 para o treinamento final completo.
    DATASET_FRACTION = 0.5

    # Hiperparâmetros de Treinamento
    BATCH_SIZE = 128
    EPOCHS = 30
    OPTIMIZER = 'SGD'
    LEARNING_RATE = 0.001 # Reduzimos a taxa de aprendizado de 0.1 para 0.01
    MOMENTUM = 0.9


    # Caminhos de Saída
    CHECKPOINT_PATH = 'models/face_recognition_resnet50_cosface.keras'
    LOG_PATH = 'deliverables/face_recognition_log.csv'
