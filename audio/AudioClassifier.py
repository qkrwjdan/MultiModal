from abc import *
import librosa
from keras.models import load_model
import os
import keras
import tensorflow as tf
import numpy as np
    

class Classifier(metaclass=ABCMeta):
    # input으로 어떤 session을 사용할 지 입력해주어야 합니다.
    def __init__(self):
        pass

    # 저장된 Keras나 TF, Pytorch 모델을 불러오는 코드. 반드시 구현 필요
    @abstractmethod
    def load_model(self, model):
        pass

    # 데이터를 외부에서 불러오는 작업이 필요하다면 구현
    def load_data(self, model):
        pass

    # 데이터 전처리가 필요하다면 구현
    def preprocess_data(self):
        pass

    # 클래스를 예측하는 코드. 반드시 구현 필요
    # 리턴 값은 string 타입의 감정이 되도록 (ang, neu, hap, ...) 구현할 것. 0, 1, 2등 숫자를 리턴하면 클래스 구분 어려움
    @abstractmethod
    def predict(self):
        pass

class AudioClassifier(Classifier):
    def __init__(self,data_path):
        self.data_path = data_path

    def load_model(self,path):
        self.model = load_model(path)

    def score(self,predictions):
        positive = 0
        negative = 0
        neutral = 0

        EMOTIONS = ['Positive', 'Negative', 'Neutral']
        for i in range(len(predictions)):
            positive += predictions[i][0]
            negative += predictions[i][1]
            neutral += predictions[i][2]
        score = [positive, negative, neutral]
        index = np.argmax(score)
        return EMOTIONS[index]

    def windows(self,data, window_size):
        start = 0
        while start < len(data):
            yield start, start + window_size
            start += (window_size / 2)

    def extract_features_array(self,filename, bands=60, frames=41):
        window_size = 512 * (frames - 1)
        log_specgrams = []
        sound_clip, s = librosa.load(filename)

        for (start, end) in self.windows(sound_clip, window_size):
            if (len(sound_clip[int(start):int(end)]) == int(window_size)):
                signal = sound_clip[int(start):int(end)]

                melspec = librosa.feature.melspectrogram(signal, n_mels=bands)
                logspec = librosa.amplitude_to_db(melspec)
                logspec = logspec.T.flatten()[:, np.newaxis].T
                log_specgrams.append(logspec)

        log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams), bands, frames, 1)
        features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis=3)
        for i in range(len(features)):
            features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])

        return np.array(features)


    def predict(self,script_id):

        file_path = os.path.join(self.data_path,script_id+".wav")
        feature_x = self.extract_features_array(file_path, bands=60, frames=41)
        predictions = self.model.predict(feature_x)

        # score(predictions)
        # predictions = prediction[0]
        # ind = np.argpartition(predictions[0], -2)[-2:]
        # ind[np.argsort(predictions[0][ind])]
        # ind = ind[::-1]
        # print "Actual:", actual, " Top guess: ", EMOTIONS[ind[0]], " (",round(predictions[0,ind[0]],3),")"
        # print "2nd guess: ", EMOTIONS[ind[1]], " (",round(predictions[0,ind[1]],3),")"
        # print(predictions)
        # index = np.argmax(predictions)
        # print("-------------------------------------------------------")
        # print(filename[:-4] + " Predicted : " + score(predictions))
        # print("=======================================================\n")
        return self.score(predictions)