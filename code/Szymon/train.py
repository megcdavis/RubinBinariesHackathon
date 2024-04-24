import argparse

import numpy as np
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

from utils import get_train_data
from models import get_cnn, get_transformer


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--tag', dest='tag', help='tag, added as a suffix to the experiment name')
parser.add_argument('--test', dest='is_test', help='Flag on test running', action='store_true')
args = parser.parse_args()

model_name = 'cnn'  # att, cnn
batch_size = 64
early_stopping = 20
epochs = 200 if not args.is_test else 2
n = None if not args.is_test else 1000

exp_name = '{}_{}'.format(model_name, args.tag)

if model_name == 'att':
    model = get_transformer(
        input_shape=(3000, 1),
        head_size=256,  # 256
        num_heads=1,  # 4
        ff_dim=4,
        num_transformer_blocks=1,  # 4
        mlp_units=[128],
        mlp_dropout=0.1,  # 0.4
        dropout=0.1,  # 0.25
    )
else:
    model = get_cnn()

callbacks = [
    ModelCheckpoint(
        filepath='./outputs/models/{}'.format(exp_name),
        save_weights_only=True,
        save_best_only=True,
        verbose=1,
    ),
    EarlyStopping(patience=early_stopping, restore_best_weights=True),
    TensorBoard(log_dir='./outputs/tensorboard/{}'.format(exp_name)),
]

X_train, X_test, y_train, y_test = get_train_data(n)

model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    shuffle=True,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=callbacks,
)

file_name = '/home/sjnakoneczny/workspace/lsst_binaries/outputs/preds/preds_{}.npy'.format(exp_name)
np.save(file_name, model.predict(X_test))
