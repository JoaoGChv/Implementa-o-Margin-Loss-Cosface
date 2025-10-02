'''# src/backbones/vgg8.py
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation, Input, MaxPooling2D, Dropout, Flatten
from tensorflow.keras import regularizers
from src.losses.margin_losses import ArcFace, CosFace, SphereFace
weight_decay = 1e-4

def vgg_block(x, filters, layers):
    for _ in range(layers):
        x = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    return x

def vgg8(args):
    # Esta função agora espera um objeto ou dict 'args' com 'num_features'
    num_features = getattr(args, 'num_features', getattr(args, 'NUM_FEATURES', 3))

    input_layer = Input(shape=(28, 28, 1))
    x = vgg_block(input_layer, 16, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = vgg_block(x, 32, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = vgg_block(x, 64, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(num_features, kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    output = Dense(10, activation='softmax', kernel_regularizer=regularizers.l2(weight_decay))(x)
    return Model(input_layer, output)

def vgg8_arcface(args):
    num_features = getattr(args, 'num_features', getattr(args, 'NUM_FEATURES', 3))

    input_image = Input(shape=(28, 28, 1))
    input_label = Input(shape=(10,))

    x = vgg_block(input_image, 16, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = vgg_block(x, 32, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = vgg_block(x, 64, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(num_features, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    output = ArcFace(10, regularizer=regularizers.l2(weight_decay))([x, input_label])
    return Model([input_image, input_label], output)

def vgg8_cosface(args):
    num_features = getattr(args, 'num_features', getattr(args, 'NUM_FEATURES', 3))

    input_image = Input(shape=(28, 28, 1))
    input_label = Input(shape=(10,))

    x = vgg_block(input_image, 16, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = vgg_block(x, 32, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = vgg_block(x, 64, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(num_features, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    output = CosFace(10, regularizer=regularizers.l2(weight_decay))([x, input_label])
    return Model([input_image, input_label], output)
'''