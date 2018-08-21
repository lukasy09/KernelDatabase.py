import numpy as np
import librosa
import re
import os
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from keras.models import  load_model


class AudioDataGenerator:
    dir_train = "./digit_dataset/train"
    num_classes = 10  # Digits [0-9]

    def __init__(self):
        self.hop_length = 512
        self.timeseries_length_list = []

    def preprocessSingleFile(self, file):
        timeseries_length = 6
        hop_length = 512
        data = np.zeros((1, timeseries_length, 33), dtype=np.float64)

        y, sr = librosa.load(file)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=13)
        spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)

        data[0, :, 0:13] = mfcc.T[0:timeseries_length, :]
        data[0, :, 13:14] = spectral_center.T[0:timeseries_length, :]
        data[0, :, 14:26] = chroma.T[0:timeseries_length, :]
        data[0, :, 26:33] = spectral_contrast.T[0:timeseries_length, :]

        return data

    def oneHotEncode(self, num_classes):
        encoder = LabelEncoder()

        self.train_Y = encoder.fit_transform(self.train_Y)
        self.train_Y = to_categorical(self.train_Y, num_classes)

    def path_to_audiofiles(self, dir_folder):
        list_of_audio = []
        for file in os.listdir(dir_folder):
            if file.endswith(".au") or file.endswith(".wav") or file.endswith(".wma"):
                directory = "%s/%s" % (dir_folder, file)
                list_of_audio.append(directory)
        return list_of_audio

    def completePreprocessAudio(self):
        trainfiles_list = self.path_to_audiofiles(self.dir_train)

        # Whole dataset.
        self.train_X, self.train_Y = self.extractAudioFeatures(trainfiles_list)

        self.oneHotEncode(self.num_classes)

    def extractAudioFeatures(self, list_of_audiofiles):
        timeseries_length = 6
        hop_length = 512
        data = np.zeros((len(list_of_audiofiles), timeseries_length, 33), dtype=np.float64)
        target = []

        for i, file in enumerate(list_of_audiofiles):
            y, sr = librosa.load(file)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=13)
            spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)

            genre = file[22]
            target.append(genre)

            data[i, :, 0:13] = mfcc.T[0:timeseries_length, :]
            data[i, :, 13:14] = spectral_center.T[0:timeseries_length, :]
            data[i, :, 14:26] = chroma.T[0:timeseries_length, :]
            data[i, :, 26:33] = spectral_contrast.T[0:timeseries_length, :]
            print("File:", file, " ", i + 1, " of ", len(list_of_audiofiles))

        print("All dataset features have been extracted.")
        return data, np.expand_dims(np.asarray(target), axis=1)
