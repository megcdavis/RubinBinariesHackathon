import keras
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten
from keras import layers


def get_transformer(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format='channels_last')(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation='relu')(x)
        x = layers.Dropout(mlp_dropout)(x)
    # outputs = layers.Dense(n_classes, activation='softmax')(x)
    outputs = layers.Dense(1)(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'],
    )

    return model


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs
        
    # return x

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res


def get_cnn():
    input_sequence = keras.Input(shape=(3000, 1))

    x = Conv1D(8, 3, activation='relu', padding='same')(input_sequence)
    x = MaxPooling1D(2, padding='same')(x)
    
    x = Conv1D(8, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2, padding='same')(x)
    
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2, padding='same')(x)
    
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2, padding='same')(x)
    
    x = Conv1D(32, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2, padding='same')(x)
    
    x = Conv1D(32, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2, padding='same')(x)
    
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2, padding='same')(x)
    
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2, padding='same')(x)
    
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2, padding='same')(x)
    
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2, padding='same')(x)
    
    x = Conv1D(256, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2, padding='same')(x)
    
    x = Conv1D(256, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2, padding='same')(x)

    x = Flatten()(x)

    x = Dense(256)(x)
    x = Dense(128)(x)

    outputs = Dense(2, activation='softmax')(x)

    model = keras.Model(input_sequence, outputs)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'],
    )

    return model
