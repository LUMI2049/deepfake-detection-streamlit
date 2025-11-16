"""
Model Architecture: EfficientNetB7 with Attention Mechanism
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras import layers, regularizers
from typing import Tuple


def attention_block(features, depth):
    """Attention mechanism for enhanced feature extraction

    Args:
        features: Input feature tensor from backbone
        depth: Depth of the feature tensor

    Returns:
        Global average pooled features with attention weights applied
    """
    attn = layers.Conv2D(256, (1, 1), padding='same', activation='relu')(layers.Dropout(0.5)(features))
    attn = layers.Conv2D(128, (1, 1), padding='same', activation='relu')(attn)
    attn = layers.Conv2D(128, (1, 1), padding='same', activation='relu')(attn)
    attn = layers.Conv2D(1, (1, 1), padding='valid', activation='sigmoid')(attn)

    up = layers.Conv2D(depth, (1, 1), padding='same', activation='linear', use_bias=False)
    up_w = np.ones((1, 1, 1, depth), dtype=np.float32)
    up.build((None, None, None, 1))
    up.set_weights([up_w])
    up.trainable = True

    attn = up(attn)
    masked = layers.Multiply()([attn, features])

    gap_feat = layers.GlobalAveragePooling2D()(masked)
    gap_mask = layers.GlobalAveragePooling2D()(attn)
    gap = layers.Lambda(lambda x: x[0] / x[1], name='RescaleGAP')([gap_feat, gap_mask])
    return gap


def build_effatt_model(input_shape: Tuple[int, int, int] = (128, 128, 3)) -> keras.Model:
    """Build EfficientNetB7 with Attention mechanism

    Args:
        input_shape: Input image shape (height, width, channels)

    Returns:
        Compiled Keras model
    """
    base_model = EfficientNetB7(include_top=False, weights=None, input_shape=input_shape)
    base_model.trainable = False

    features = base_model.output
    bn_features = layers.BatchNormalization()(features)
    pt_depth = base_model.output_shape[-1]
    gap = attention_block(bn_features, pt_depth)

    x = layers.Dropout(0.5)(gap)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.00001))(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(2, activation='softmax')(x)

    return keras.Model(inputs=base_model.input, outputs=outputs)
