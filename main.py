import os
import librosa  # for sound processing.
import numpy as np
import pandas as pd
from keras.utils import np_utils
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras import regularizers, optimizers
import config


def CNN(input_shape):
    from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
    from keras.models import Sequential, Model
    from keras.layers import Conv2D, MaxPooling2D

    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same',
                     input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 4)))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Conv2D(4, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return model

def calculate_mel_spec(audio, transpose=True, expand_dims=(0, )):
    """
    Calculate a mal spectrogram from raw audio waveform
    Note: The parameters of the spectrograms are in the config.py file.
    Args:
        audio : numpy.array, raw waveform to compute the spectrogram

    Returns:
        numpy.array
        containing the mel spectrogram
    """
    # Compute spectrogram
    ham_win = np.hamming(cfg.n_window)

    spec = librosa.stft(
        audio,
        n_fft=n_window,
        hop_length=hop_length,
        window=ham_win,
        center=True,
        pad_mode='reflect'
    )

    mel_spec = librosa.feature.melspectrogram(
        S=np.abs(spec),  # amplitude, for energy: spec**2 but don't forget to change amplitude_to_db.
        sr=cfg.sr,
        n_mels=cfg.n_mels,
        fmin=cfg.f_min, fmax=cfg.f_max,
        htk=False, norm=None)

    if save_log_feature:
        mel_spec = librosa.amplitude_to_db(mel_spec)  # 10 * log10(S**2 / ref), ref default is 1

    if transpose:
        mel_spec = mel_spec.T

    if len(expand_dims) > 0:
        mel_spec = np.expand_dims(mel_spec, axis=expand_dims)

    return mel_spec


def pad_zeros(array, size, axis):
    shape = list(array.shape)

    if shape[axis] > size:
        shape[axis] = size
        shape = tuple(slice(0, i) for i in shape)
        return array[shape]

    shape[axis] = size - shape[axis]
    return np.concatenate((array, np.zeros(shape)), axis=axis)


def train(cfg):
    data_path = cfg.data_path
    if data_path[-1] == '/':
        data_path = data_path[:-1]
    csv_path = data_path + "/metadata/UrbanSound8K.csv"
    data = pd.read_csv(csv_path)
    train_idx, test_idx = train_test_split(data.index, test_size=0.3, stratify=data['class'])
    valid_idx, test_idx = train_test_split(test_idx, test_size=0.33, stratify=data.loc[test_idx]['class'])

    # Split data on train / valid / test
    data.at[train_idx, 'split'] = 'train'
    data.at[valid_idx, 'split'] = 'valid'
    data.at[test_idx, 'split'] = 'test'

    if '.' in csv_path:
        data.to_csv(csv_path[:-4] + '_split' + '.csv')
    else:
        data.to_csv(csv_path + '_split' + '.csv')

    # Read audio files and generate features
    if 'fold' in data.columns and \
            sum([a.startswith("fold") for a in next(os.walk(data_path))[1]]) > 2:
        data['audio'] = data[['slice_file_name', 'fold']].apply(lambda x: librosa.load(data_path + "/fold{}/".format(x.fold) + x.slice_file_name, sr=cfg.sr)[0], axis=1)

    else:
        data['audio'] = data['slice_file_name'].apply(lambda x: librosa.load(data_path + '/' + x, sr=cfg.sr)[0])

    data['features'] = data['audio'].apply(calculate_mel_spec)

    # Pad features to the same shape
    if cfg.pad_size is None:
        pad_size = data['features'].apply(lambda x: x.shape[1]).max()
    else:
        pad_size = cfg.pad_size

    data['features'] = data['features'].apply(lambda x: pad_zeros(x, pad_size, axis=1))

    X_train, X_valid, X_test = (np.array(data[data.split == sp]['features'].tolist())
                                for sp in ['train', 'valid', 'test'])

    lb = LabelEncoder().fit(cfg.classes)
    y_train, y_valid, y_test = (
        np_utils.to_categorical(lb.transform(data[data.split == sp]['class'].tolist()), num_classes=len(lb.classes_))
        for sp in ['train', 'valid', 'test'])

    # Train model
    model = CNN(X_train.shape[1:])
    model.compile(optimizers.RMSprop(lr=0.0005, decay=1e-6), loss="categorical_crossentropy", metrics=["accuracy"])
    print(model.summary())
    model.fit(X_train, y_train, batch_size=64, epochs=150, validation_data=(X_valid, y_valid))

    if cfg.save_path:
        with open(cfg.save_path + "/model.json", "w") as json_file:
            json_file.write(model.to_json())
        model.save_weights(cfg.save_path + "/model.h5")

        # json_file = open('model.json', 'r')
        # loaded_model_json = json_file.read()

    # Test model quality
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    print(classification_report(y_test, y_pred, target_names=lb.classes_))


def evaluate(cfg):
    data_path = cfg.data_path
    if data_path[-1] == '/':
        data_path = data_path[:-1]
    if os.path.exists(data_path[:-4] + '_split_' + '.csv'):
        data = pd.read_csv(data_path[:-4] + '_split_' + '.csv')
        data = data[data.split == 'test']
    elif os.path.exists(data_path + '_split_' + '.csv'):
        data = pd.read_csv(data_path + '_split_' + '.csv')
        data = data[data.split == 'test']
    else:
        data = pd.read_csv(data_path)

    # Read audio files and generate features
    if 'fold' in data.columns() and \
            sum([a.startswith("fold") for a in next(os.walk("UrbanSound8K"))[1]]) > 2:
        data['audio'] = data[['slice_file_name', 'fold']].apply(
            lambda x: librosa.load("UrbanSound8K/" + "fold{}/".format(x.fold) + x.slice_file_name, sr=cfg.sr))

    else:
        data['audio'] = data['slice_file_name'].apply(lambda x: librosa.load(x, sr=cfg.sr))

    data['features'] = data['audio'].apply(calculate_mel_spec)
    X = np.array(data['features'].apply(lambda x: pad_zeros(x, pad_size, axis=1)).tolist())

    lb = LabelEncoder().fit(cfg.classes)
    y = np_utils.to_categorical(lb.transform(np.array(data['class'].tolist())), num_classes=len(lb.classes_))

    with open(cfg.save_path + "/model.json", 'r') as json_file:
        model = json_file.read()

    print(classification_report(y, model.predict(X), target_names=lb.classes_))


def predict(cfg, return_audio=False):
    audio, _ = librosa.load(cfg.filename, sr=cfg.sr)
    features = calculate_mel_spec(audio)
    features = pad_zeros(features, cfg.pad_size, cfg.pad_axis)
    with open(cfg.save_path + "/model.json", 'r') as json_file:
        model = json_file.read()
    pred_label = lb.inverse_transform([np.argmax(model(features))])
    if return_audio:
        return pred_label, audio
    else:
        return pred_label

if __name__ == '__main__':
    train(config)