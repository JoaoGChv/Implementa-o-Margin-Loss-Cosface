# Local: src/pipelines/face_training_pipeline.py

import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TerminateOnNaN, EarlyStopping

# Importa os módulos do nosso projeto
from src.backbones.resnet import create_resnet50_cosface
from src.data_loader.face_datasets import get_train_val_datasets, LFWValidationCallback
from src.optimizers.scheduler import CosineAnnealingScheduler

def plot_training_history(log_data, save_path):
    """Plota e salva as curvas de acurácia e perda a partir de um DataFrame."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot de Acurácia (no dataset de treino/validação do VGGFace2)
    ax1.plot(log_data['epoch'], log_data['accuracy'], label='Treino Acurácia')
    ax1.plot(log_data['epoch'], log_data['val_accuracy'], label='Validação Acurácia')
    ax1.set_title('Histórico de Acurácia (VGGFace2)')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Acurácia')
    ax1.legend()

    # Plot de Perda
    ax2.plot(log_data['epoch'], log_data['loss'], label='Treino Perda')
    ax2.plot(log_data['epoch'], log_data['val_loss'], label='Validação Perda')
    ax2.set_title('Histórico de Perda (VGGFace2)')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Perda')
    ax2.legend()

    fig.tight_layout()
    plt.savefig(os.path.join(save_path, 'face_training_history.png'))
    print(f"Gráfico do histórico de treino salvo em: {save_path}")

def plot_learning_rate(log_data, save_path):
    """Plota e salva o histórico da taxa de aprendizado."""
    if 'lr' in log_data.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(log_data['epoch'], log_data['lr'], label='Taxa de Aprendizagem')
        plt.title('Agendamento da Taxa de Aprendizagem')
        plt.xlabel('Época')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.savefig(os.path.join(save_path, 'face_learning_rate.png'))
        print(f"Gráfico da taxa de aprendizado salvo em: {save_path}")

def run_face_training(config, vggface2_path, lfw_path, lfw_pairs_path):
    """Executa o pipeline completo de treinamento em duas fases."""
    print("--- Iniciando Pipeline de Treinamento de Faces ---")

    FIGURES_PATH = 'reports/figures'
    os.makedirs(FIGURES_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(config.CHECKPOINT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(config.LOG_PATH), exist_ok=True)

    # 1. Carregar Dados
    train_dataset, val_dataset, num_classes_in_subset = get_train_val_datasets(
        vggface2_path, config.IMAGE_SIZE, config.BATCH_SIZE, fraction=config.DATASET_FRACTION
    )
    config.NUM_CLASSES = num_classes_in_subset

    # 2. Criar o Modelo
    model, feature_extractor = create_resnet50_cosface(config)

    # === FASE 1: TREINAR APENAS A CABEÇA ===
    print("\n--- FASE 1: Treinando apenas a cabeça do modelo ---")

    # Compila o modelo com o backbone congelado
    optimizer_head = Adam(learning_rate=1e-3) # Adam é bom para iniciar
    model.compile(loss='categorical_crossentropy', optimizer=optimizer_head, metrics=['accuracy'])

    model.fit(train_dataset.take(2000), # Usa um subconjunto pequeno para aquecer a cabeça
              epochs=5,
              validation_data=val_dataset.take(200),
              verbose=1)

    # === FASE 2: FINE-TUNING DO MODELO COMPLETO ===
    print("\n--- FASE 2: Iniciando fine-tuning do modelo completo ---")

    # Descongela as camadas finais do backbone para fine-tuning
    for layer in model.get_layer('resnet50').layers[-46:]:
        layer.trainable = True

    # Compila o modelo novamente com uma taxa de aprendizado muito baixa
    optimizer_finetune = SGD(learning_rate=config.LEARNING_RATE, momentum=config.MOMENTUM)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer_finetune, metrics=['accuracy'])

    print("Sumário do modelo para fine-tuning:")
    model.summary()

    # Callbacks para a fase de fine-tuning
    checkpoint_path_h5 = config.CHECKPOINT_PATH.replace('.keras', '.h5')
    weights_checkpoint_path_h5 = checkpoint_path_h5.replace('.h5', '.weights.h5')
    lfw_callback = LFWValidationCallback(feature_extractor, lfw_path, lfw_pairs_path, config.IMAGE_SIZE)
    early_stopping_callback = EarlyStopping(monitor='val_lfw_accuracy', patience=5, mode='max', restore_best_weights=True)

    callbacks = [
        ModelCheckpoint(checkpoint_path_h5, monitor='val_lfw_accuracy', mode='max', verbose=1, save_best_only=True),
        ModelCheckpoint(weights_checkpoint_path_h5, monitor='val_lfw_accuracy', mode='max', verbose=1, save_best_only=True, save_weights_only=True),
        CSVLogger(config.LOG_PATH),
        lfw_callback,
        early_stopping_callback
    ]

    # Treinamento de fine-tuning
    history = model.fit(train_dataset,
              epochs=config.EPOCHS,
              callbacks=callbacks,
              validation_data=val_dataset,
              verbose=1)

    print("\n--- Treinamento Concluído ---")

    # 6. Análise Pós-Treino
    print("\n--- Iniciando Análise Pós-Treino ---")
    if os.path.exists(config.LOG_PATH):
        log_data = pd.read_csv(config.LOG_PATH)
        plot_training_history(log_data, FIGURES_PATH)
        plot_learning_rate(log_data, FIGURES_PATH)
    else:
        print(f"AVISO: Arquivo de log não encontrado em {config.LOG_PATH}. Pulando geração de gráficos.")