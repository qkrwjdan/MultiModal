import os
from libs.model import TextClassifier, VideoClassifier #, AudioClassifier

class Voter:

    def __init__(self, test_session_no=[1], include_neu=False, text_weight=0.71, video_weight=0.72, audio_weight=0.7):
        # weight : accuracy
        self._text_weight = text_weight
        self._video_weight = video_weight
        self._audio_weight= audio_weight
        self._label_idx = {'Positive':0, 'Negative':1,'Neutral':2}
        self._include_neu = include_neu
        if include_neu:
            self._score_board = {'Positive':0, 'Negative':0,'Neutral':0}
        else:
            self._score_board = {'Positive':0, 'Negative':0,'Neutral':0}
        
    def scoring(self, pred_class, weight):
        self._score_board[pred_class] += weight

    def decide_emotion(self):
        emotion = max(self._score_board, key=self._score_board.get)
        print(self._score_board)
        self.reset_score_board()
        return emotion
    
    def reset_score_board(self):
        if self._include_neu:
            self._score_board = {'Positive':0, 'Negative':0,'Neutral':0}
        else:
            self._score_board = {'Positive':0, 'Negative':0,'Neutral':0}


    def voting(self, test_id):
        # test_id = 'Ses01F_impro01_F012'

        t = TextClassifier()
        t.load_model(model_path='models/text/class3_model.h5', tokenizer_path='models/text/class3_tokenizer.pickle', le_path='models/text/class3_label_encoder.pickle')
        text_predict = t.predict(test_id)

        audio_fname = f"{test_id}.wav"
        print(audio_fname)
        #audio_path = os.path.join('dataset', 'iemocap_audio', 'raw', audio_fname)
        audio_path = os.path.join(os.getcwd(),'dataset\iemocap_audio\\raw', audio_fname)
        print(audio_path)
        a = AudioClassifier(audio_path)
        a.load_model('models/audio/cnn_session1_2_3_test.h5')
        audio_predict = a.predict()

        v = VideoClassifier(test_id,session_nums=[1],include_neu=False)
        v.preprocess_data()
        v.load_model('models/video/3class_session1_2_3')
        video_predict = v.predict()

        print(text_predict, self._text_weight)
        self.scoring(text_predict, self._text_weight)
        self.scoring(video_predict, self._video_weight)
        self.scoring(audio_predict, self._audio_weight)

        emotion = self.decide_emotion()

        return emotion
