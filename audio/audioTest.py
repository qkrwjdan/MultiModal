import pandas as pd
import numpy as np
import random
import os
import time
from AudioClassifier import AudioClassifier

master = pd.read_csv(os.path.join(os.getcwd(),'datasets','master','raw_session1.csv'))
master = master[['Clip_Name','Use','Label','Emotion']]

master_test = master[master['Use']=='test']

print(master_test)
audio_path = os.path.join('datasets', 'iemocap_audio', 'raw')
print(audio_path)

a = AudioClassifier(audio_path)
a.load_model('models/audio/crnn_session1_5_class3.h5')

start = time.time()

ans = 0
all = 0

for test_id,label in zip(master_test["Clip_Name"],master_test["Label"]):
    print(test_id)

    pred = a.predict(test_id)
    print("pred : ",pred)
    if pred == label:
        ans += 1
    all += 1

print("ACC : ",ans / all * 100)
print("time : ",time.time() - start)

# a = AudioClassifier(audio_path)
# a.load_model('models/audio/cnn_session1_2_3_test.h5')
# audio_predict = a.predict()