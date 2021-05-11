import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import RobertaModel, RobertaTokenizer

import os
import random

from abc import *

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
            if include_neu:
                fname = f'session{session_nums[0]}_text_neu.csv'
            else:
                fname = f'session{session_nums[0]}_text.csv'
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
        emotion = row['Label']

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