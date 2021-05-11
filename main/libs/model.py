import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Error만 출력
import pandas as pd
import keras
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
# import nltk

#text
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import RobertaModel, RobertaTokenizer

#video
import tensorflow as tf
from keras import utils
from skimage.io import imread
from skimage.transform import resize


from abc import *
import librosa
from keras.models import load_model
    

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

class BERT_Arch(nn.Module):
    
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        
        self.bert = bert
        
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768,512)
        self.fc2 = nn.Linear(512,3)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, sent_id, mask):
        
        ret = self.bert(sent_id, attention_mask = mask).pooler_output
#         ret = self.bert(sent_id, attention_mask = mask)
        
#         print(ret)
        
        x = self.fc1(ret)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        
#         print(x.size())
        return x

class TextClassifier(Classifier):

    def __init__(self, session_nums=[1,2,3], base_dir='.', include_neu=False):
        """입력받은 값을 검증하고 필요한 데이터 및 라이브러리를 로드합니다.
        
        Arguments:
            session_nums {list} -- [실험에 사용할 세션을 지정합니다.]
        
        Keyword Arguments:
            base_dir {str} -- [작업 공간을 지정합니다.] (default: {'.'})
            include_neu {bool} -- [참조할 데이터셋에 중립 감정을 포함할지 설정합니다.] (default: {True})
        
        Raises:
            TypeError: session_nums 파라미터는 반드시 리스트로 받아야 합니다.
            Exception: []
        """
        if torch.cuda.is_available():
            print(torch.cuda.get_device_name(0))
            self.device = torch.device("cuda")
        else: 
            self.device = torch.device("cpu")

        self.labelEncoder = {"Negative" : 1, "Neutral" : 2, "Positive" : 0}
        self.labelDecoder = {0 : "Positive",1 : "Negative", 2:"Neutral"} 

        if not isinstance(session_nums, list):
            raise TypeError('session no must be list type')
        else:
            self.session_nums = session_nums

        # for session_num in session_nums:
        #     self.make_text_dataset(session_num, include_neu=include_neu)

        # base_dir = os.path.join('drive', 'My Drive', 'chatbot')
        text_dir = os.path.join(base_dir, 'datasets',  'iemocap_text')

        if len(session_nums) == 1:
            # if include_neu:
            #     fname = f'session{session_nums[0]}_text_neu.csv'
            # else:
            #     fname = f'session{session_nums[0]}_text.csv'
            fname = f'session{session_nums[0]}_text_neu.csv'
            self.text_dset = pd.read_csv(os.path.join(text_dir, f'session{session_nums[0]}_text.csv'))
        elif len(session_nums) > 1:
            if include_neu:
                dir_list = [os.path.join(text_dir, f'session{no}_text_neu.csv') for no in session_nums]
            else:
                dir_list = [os.path.join(text_dir, f'session{no}_text.csv') for no in session_nums]
            base = pd.read_csv(dir_list[0])
            for path in dir_list[1:]:
                df = pd.read_csv(path)
                base = base.append(df, ignore_index=True)
            
            self.text_dset = base
        else:
            raise Exception('Unknown Error')
        
        # 필요한 nltk 라이브러리 다운
        # nltk.download('popular')
        # nltk.download('stopwords')

    def load_model(self,
                base_model = "roberta-base",
                model_path='models/text/emocap_3class_rbt.pt',
                max_length = 30
                ):
        """pre-train 된 모델을 불러옵니다.
        
        Keyword Arguments:
            base_model {str} -- pre-trained model 
            model_path {str} -- 불러올 모델이 저장된 경로를 입력합니다. (default: {'models/text_model.h5'})
        
        """
        self.max_length = max_length

        bert = RobertaModel.from_pretrained(base_model)
        self.tokenizer = RobertaTokenizer.from_pretrained(base_model)
        self.model = BERT_Arch(bert)
        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.to(self.device)

    def get_data(self, script_id):
        dset = self.text_dset
        row = dset[dset['Clip_Name'] == script_id]
        text = row['text']
        if text.empty :
            text = "empty"

        emotion = row['Label']
        print("text : {text}".format(text=text))
        return text, emotion

    def preprocess(self, text):
        
        tokenized = self.tokenizer.batch_encode_plus(
            text,
            max_length = self.max_length,
            pad_to_max_length = True,
            truncation = True
        )

        seq = torch.tensor(tokenized['input_ids'])
        mask = torch.tensor(tokenized['attention_mask'])
        return (seq,mask)


    def predict(self, script_id):
        
        text, _ = self.get_data(script_id)
        seq, mask = self.preprocess(text)

        with torch.no_grad():
            pred = self.model(seq.to(self.device),mask.to(self.device))
            pred = pred.detach().cpu().numpy()

        pred = np.argmax(pred,axis = 1)

        return self.labelDecoder[pred[0]]


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
        self.work_directory = self.work_directory + '/datasets/'

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