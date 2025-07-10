import tensorflow as tf
from tensorflow.keras import layers, models

def residual_block(inputs, filters, kernel_size=3, strides=1):
    """Define residual block for the SegResNet architecture"""
    x = layers.Conv3D(filters, kernel_size, strides=strides, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    
    x = layers.Conv3D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    if strides > 1 or inputs.shape[-1] != filters:
        skip = layers.Conv3D(filters, 1, strides=strides, padding='same')(inputs)
        skip = layers.BatchNormalization()(skip)
    else:
        skip = inputs
    
    x = layers.Add()([x, skip])
    x = layers.LeakyReLU(alpha=0.01)(x)
    return x

def build_segresnet(input_shape=(128, 128, 96, 4), num_classes=4):
    """Build the SegResNet model"""
    inputs = layers.Input(input_shape)
    
    # Initial convolution
    x = layers.Conv3D(32, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    
    # Encoder path
    skip_connections = []
    
    # Encoder block 1
    x = residual_block(x, 32)
    skip_connections.append(x)
    x = layers.MaxPooling3D(pool_size=2)(x)
    
    # Encoder block 2
    x = residual_block(x, 64)
    skip_connections.append(x)
    x = layers.MaxPooling3D(pool_size=2)(x)
    
    # Encoder block 3
    x = residual_block(x, 128)
    skip_connections.append(x)
    x = layers.MaxPooling3D(pool_size=2)(x)
    
    # Bridge
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    
    # Decoder path
    skip_connections = skip_connections[::-1]  # Reverse the list for decoder
    
    # Decoder block 1
    x = layers.UpSampling3D(size=2)(x)
    x = layers.Concatenate()([x, skip_connections[0]])
    x = residual_block(x, 128)
    
    # Decoder block 2
    x = layers.UpSampling3D(size=2)(x)
    x = layers.Concatenate()([x, skip_connections[1]])
    x = residual_block(x, 64)
    
    # Decoder block 3
    x = layers.UpSampling3D(size=2)(x)
    x = layers.Concatenate()([x, skip_connections[2]])
    x = residual_block(x, 32)
    
    # Output layer
    outputs = layers.Conv3D(num_classes, 1, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model