import os 
import tensorflow as tf
import keras
import pandas as pd
import numpy as np

from abc import *

from keras import utils
from skimage.io import imread
from skimage.transform import resize

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

class Video_Custom_test_Generator(utils.Sequence):

    def __init__(self, image_filenames, labels, batch_size, image_path):
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.image_path = image_path

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]
        return np.array([
            resize(imread(self.image_path + str(file_name), plugin='matplotlib'),(224,224,3))
            for file_name in batch_x]) / 255.0, np.array(batch_y)

class VideoClassifier(Classifier):
    def __init__(self, session_nums=[1, 2, 3], include_neu=False):

        # 현재 작업 폴더
        self.work_directory = os.getcwd()
        self.work_directory = self.work_directory + '/dataset/'

        # 자료들이 담길 폴더 생성
        self.temp_directory = self.work_directory + 'iemocap_video/image_model/'
        if not os.path.isdir(self.temp_directory):
            os.makedirs(self.work_directory + 'iemocap_video/image_model')

        # 이미지 파일들 경
        self.image_directory = self.temp_directory + 'test_1_clip/'

        # npy파일들 경로
        self.npy_directory = self.temp_directory + 'test_1_clip_npy/'

        self.target_clip_emotion = ''

        self.include_neu = include_neu

    def find_max(self, y_list):
        maxValue = y_list[0]
        max_i = 0
        for i in range(1, len(y_list)):
            if maxValue < y_list[i]:
                maxValue = y_list[i]
                max_i = i
        return max_i

    def load_model(self, model_path):

        from keras import optimizers

        json_file = open(model_path + ".json", "r")
        loaded_model_json = json_file.read()
        json_file.close()

        self.model = keras.models.model_from_json(loaded_model_json)
        self.model.load_weights(model_path + ".h5")
        self.model.compile(loss='binary_crossentropy',
                           optimizer=optimizers.SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True),
                           metrics=['accuracy'])

    def predict(self,video_name):

        self.target_clip_name = video_name

        X_test_filenames = np.load(self.npy_directory + self.target_clip_name + '_filename.npy')
        y_test = np.load(self.npy_directory + self.target_clip_name + '_label.npy')

        my_test_batch_generator = Video_Custom_test_Generator(X_test_filenames, y_test, 1, self.image_directory)
        predict_list =  self.model.predict_generator(my_test_batch_generator, steps=len(X_test_filenames))

        y_frame = []
        y_label = [0, 0, 0]
        for k in range(len(predict_list)):
            y_frame.append(self.find_max(predict_list[k]))
            y_label[self.find_max(predict_list[k])] += 1
        max = self.find_max(y_label)
        if max == 0:
            self.target_clip_emotion = "Positive"
        elif max == 1:
            self.target_clip_emotion = "Negative"
        elif max == 2:
            self.target_clip_emotion = "Neutral"

        return self.target_clip_emotion
