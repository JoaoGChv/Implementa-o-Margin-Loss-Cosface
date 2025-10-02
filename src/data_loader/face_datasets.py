
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
import random

# Camadas de Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

def get_train_val_datasets(path, image_size, batch_size, validation_split=0.02, fraction=1.0):
    """
    Cria conjuntos de dados de treino e validação a partir do VGGFace2,
    usando uma fração das classes e aplicando data augmentation de forma eficiente.
    """
    
    # 1. Seleciona a fração de classes a serem usadas
    all_class_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    random.shuffle(all_class_dirs)
    num_classes_to_use = int(len(all_class_dirs) * fraction)
    selected_class_dirs = all_class_dirs[:num_classes_to_use]
    print(f"Usando uma fração de {fraction*100:.1f}% do dataset: {num_classes_to_use} de {len(all_class_dirs)} classes.")

    # 2. Cria listas de caminhos de arquivo e RÓTULOS INTEIROS (memory-efficient)
    file_paths = []
    labels = []
    label_to_int = {name: i for i, name in enumerate(selected_class_dirs)}

    for class_name in selected_class_dirs:
        class_dir = os.path.join(path, class_name)
        for fname in os.listdir(class_dir):
            file_paths.append(os.path.join(class_dir, fname))
            labels.append(label_to_int[class_name])
    
    # 3. Cria o dataset a partir das listas
    path_ds = tf.data.Dataset.from_tensor_slices(file_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels) # Usamos rótulos inteiros aqui
    image_label_ds = tf.data.Dataset.zip((path_ds, label_ds))
    image_label_ds = image_label_ds.shuffle(buffer_size=len(file_paths), seed=123)
    
    # 4. Divide em treino/validação
    val_size = int(len(file_paths) * validation_split)
    train_ds = image_label_ds.skip(val_size)
    val_ds = image_label_ds.take(val_size)

    # 5. Define as funções de processamento
    def load_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, image_size)
        return img, label

    def augment_and_prepare(image, label):
        # Converte o rótulo para one-hot DENTRO do fluxo de dados
        one_hot_label = tf.one_hot(label, depth=num_classes_to_use)
        image = data_augmentation(image, training=True)
        image = tf.keras.applications.resnet50.preprocess_input(image)
        return (image, one_hot_label), one_hot_label

    def validate_and_prepare(image, label):
        # Converte o rótulo para one-hot DENTRO do fluxo de dados
        one_hot_label = tf.one_hot(label, depth=num_classes_to_use)
        image = tf.keras.applications.resnet50.preprocess_input(image)
        return (image, one_hot_label), one_hot_label

    # 6. Constrói o pipeline final
    train_ds = train_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(augment_and_prepare, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = val_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(validate_and_prepare, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds, num_classes_to_use

class LFWValidationCallback(tf.keras.callbacks.Callback):
    """Callback para validar o modelo no LFW ao final de cada época."""
    def __init__(self, feature_extractor, lfw_path, pairs_path, image_size):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.lfw_path = lfw_path
        self.image_size = image_size
        self.pairs = self._read_pairs(pairs_path)

    def _read_pairs(self, pairs_filename):
        pairs = []
        with open(pairs_filename, 'r') as f:
            for line in f.readlines()[1:]: # Pula o cabeçalho
                pair = line.strip().split()
                pairs.append(pair)
    
        return pairs

    def _preprocess_image(self, file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, self.image_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img

    def on_epoch_end(self, epoch, logs=None):
        print("\nIniciando validação no LFW...")
 

        embeddings1 = []
        embeddings2 = []
        actual_issame = []

        for pair in self.pairs:
            if len(pair) == 3: # Par positivo
                path1 = os.path.join(self.lfw_path, pair[0], f"{pair[0]}_{int(pair[1]):04d}.jpg")
                path2 = os.path.join(self.lfw_path, pair[0], f"{pair[0]}_{int(pair[2]):04d}.jpg")
                actual_issame.append(True)
            elif len(pair) == 4: # Par negativo
                path1 = os.path.join(self.lfw_path, pair[0], f"{pair[0]}_{int(pair[1]):04d}.jpg")
                path2 = os.path.join(self.lfw_path, pair[2], f"{pair[2]}_{int(pair[3]):04d}.jpg")
                actual_issame.append(False)

            img1 = self._preprocess_image(path1)
            img2 = self._preprocess_image(path2)

            emb1 = self.feature_extractor(tf.expand_dims(img1, axis=0), training=False)
            emb2 = self.feature_extractor(tf.expand_dims(img2, axis=0), training=False)

            embeddings1.append(emb1)
            embeddings2.append(emb2)

        emb1_tensor = tf.concat(embeddings1, axis=0)
        emb2_tensor = tf.concat(embeddings2, axis=0)

        # Calcula a distância de cosseno
        diff = tf.reduce_sum(tf.multiply(emb1_tensor, emb2_tensor), axis=1)

        # Cálculo de acurácia simples com um threshold
        threshold = 0.5 
        predict_issame = tf.greater(diff, threshold)

        correct_predictions = tf.equal(predict_issame, actual_issame)
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

        print(f"LFW Validação - Acurácia (threshold={threshold}): {accuracy.numpy():.4f}")
        logs['val_lfw_accuracy'] = accuracy.numpy()